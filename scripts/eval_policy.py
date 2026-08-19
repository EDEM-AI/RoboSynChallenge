#!/usr/bin/env python
# ----------------------------------------------------------------------------
# RoboSynChallenge — unified policy eval script
# python scripts/eval_policy.py --config policy/{policy_name}/deploy_policy.yml
# ----------------------------------------------------------------------------

import os
import sys
import argparse
import importlib
import json
import platform
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import yaml
from tqdm.auto import tqdm

# Add policy directory to path for dynamic imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
POLICY_DIR = os.path.join(REPO_ROOT, "policy")


def prepend_existing_paths(paths):
    """Prepend existing paths to sys.path while preserving the given order."""
    for path in reversed(paths):
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)


WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
EMBODICHAIN_ROOT = os.environ.get(
    "EMBODICHAIN_ROOT", os.path.join(WORKSPACE_ROOT, "EmbodiChain")
)
prepend_existing_paths([REPO_ROOT, POLICY_DIR, EMBODICHAIN_ROOT])


def add_policy_dependency_paths(policy_name):
    """Expose policy-local pure-Python source without replacing simulator Python."""
    policy_root = os.path.join(POLICY_DIR, policy_name)
    paths = [
        policy_root,
        os.path.join(policy_root, "src"),
        os.path.join(policy_root, "packages", "openpi-client", "src"),
    ]
    prepend_existing_paths(paths)

import robosynchallenge
import embodichain.lab.gym.utils.gym_utils as gym_utils
from embodichain.utils import logger as emb_logger

CHALLENGE_MANAGER_MODULES = [
    "robosynchallenge.managers.actions",
    "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events",
    "robosynchallenge.managers.observations",
]

for module in CHALLENGE_MANAGER_MODULES:
    if module not in gym_utils.DEFAULT_MANAGER_MODULES:
        gym_utils.DEFAULT_MANAGER_MODULES.append(module)


def load_policy_adapter(policy_name):
    """Dynamically import a policy adapter package.

    Each policy adapter must live at policy/<policy_name>/ and export these
    functions from its top-level namespace (via __init__.py):

        get_model(usr_args: dict) -> model
        eval(env: gym.Env, model, obs) -> (obs, info, truncated, inference_times)
        reset_model(model) -> None
    """
    add_policy_dependency_paths(policy_name)
    try:
        pkg = importlib.import_module(f"policy.{policy_name}")
    except ImportError as e:
        print(f"Error: cannot import policy '{policy_name}': {e}")
        sys.exit(1)

    for fn_name in ("get_model", "eval", "reset_model"):
        if not hasattr(pkg, fn_name):
            print(f"Error: policy '{policy_name}' missing required function '{fn_name}'")
            sys.exit(1)

    return pkg


def _set_default(config, key, value):
    if config.get(key) is None:
        config[key] = value


def _as_int_or_none(value):
    if value is None:
        return None
    return int(value)


def resolve_episode_max_steps(config, gym_config):
    """Use the larger env-step limit from deploy config and gym config."""
    deploy_max_steps = _as_int_or_none(config.get("max_steps"))
    gym_max_steps = _as_int_or_none(gym_config.get("max_episode_steps"))
    candidates = [value for value in (deploy_max_steps, gym_max_steps) if value is not None]
    max_env_steps = max(candidates) if candidates else 300
    return max_env_steps, deploy_max_steps, gym_max_steps


class EpisodeVideoRecorder:
    """Record one RGB observation stream per episode through an ffmpeg pipe."""

    def __init__(self, save_dir, obs_keys="cam_high", fps=10, crf=23):
        self.save_dir = Path(save_dir)
        if isinstance(obs_keys, str):
            obs_keys = [key.strip() for key in obs_keys.split(",") if key.strip()]
        self.obs_keys = list(obs_keys)
        if not self.obs_keys:
            raise ValueError("At least one eval video observation key is required.")
        self.fps = int(fps)
        self.crf = int(crf)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._process = None
        self._episode_idx = None
        self._episode_seed = None
        self._tmp_path = None
        self._frame_size = None
        self._frame_count = 0

    def start_episode(self, episode_idx, seed):
        self.close_episode(success=False)
        self._episode_idx = int(episode_idx)
        self._episode_seed = int(seed)
        self._tmp_path = self.save_dir / f"episode_{episode_idx:03d}_seed_{seed}.tmp.mp4"
        self._process = None
        self._frame_size = None
        self._frame_count = 0

    def record(self, obs):
        if self._episode_idx is None:
            return

        frames = [self._extract_frame(obs, obs_key) for obs_key in self.obs_keys]
        frame = np.ascontiguousarray(np.concatenate(frames, axis=1))
        height, width = frame.shape[:2]
        if self._process is None:
            self._start_writer(width, height)
        elif self._frame_size != (width, height):
            raise ValueError(
                f"Video frame size changed from {self._frame_size} to {(width, height)}."
            )

        self._process.stdin.write(frame.tobytes())
        self._frame_count += 1

    def _extract_frame(self, obs, obs_key):
        frame = obs["sensor"][obs_key]["color"]
        if frame is None:
            raise KeyError(
                f"EmbodiChain camera color observation 'sensor/{obs_key}/color' "
                "not found for video recording."
            )

        frame = np.asarray(frame[0, ..., :3])
        if frame.dtype != np.uint8:
            max_value = float(np.nanmax(frame)) if frame.size else 0.0
            if max_value <= 1.5:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def close_episode(self, success=None):
        if self._process is None:
            self._reset_episode_state()
            return

        self._process.stdin.close()
        return_code = self._process.wait()
        if return_code != 0:
            tmp_path = self._tmp_path
            self._reset_episode_state()
            raise RuntimeError(f"ffmpeg failed while saving eval video: {tmp_path}")

        if self._frame_count > 0:
            status = "success" if success else "fail"
            final_path = (
                self.save_dir
                / f"episode_{self._episode_idx:03d}_seed_{self._episode_seed}_{status}.mp4"
            )
            os.replace(self._tmp_path, final_path)
            print(f"  Video saved: {final_path}")
        elif self._tmp_path and self._tmp_path.exists():
            self._tmp_path.unlink()

        self._reset_episode_state()

    def _start_writer(self, width, height):
        self._frame_size = (width, height)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            str(self.crf),
            str(self._tmp_path),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def _reset_episode_state(self):
        self._process = None
        self._episode_idx = None
        self._episode_seed = None
        self._tmp_path = None
        self._frame_size = None
        self._frame_count = 0


class RecordingEnvProxy:
    """Proxy env.step/reset so policy adapters do not need video-specific code."""

    def __init__(self, env, recorder=None, reset_sync_steps=0):
        self._env = env
        self._recorder = recorder
        self._reset_sync_steps = int(reset_sync_steps)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs, info = self._sync_reset_obs(obs, info)
        if self._recorder:
            self._recorder.record(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._recorder and not (
            self._as_done(terminated) or self._as_done(truncated)
        ):
            self._recorder.record(obs)
        return obs, reward, terminated, truncated, info

    def _sync_reset_obs(self, obs, info):
        if self._reset_sync_steps <= 0:
            return obs, info

        env = getattr(self._env, "unwrapped", self._env)
        sim = getattr(env, "sim", None)
        if sim is None:
            return obs, info

        physics_dt = getattr(getattr(env, "sim_cfg", None), "physics_dt", None)
        sim.update(physics_dt, self._reset_sync_steps)

        if hasattr(env, "get_obs"):
            obs = env.get_obs()
        if hasattr(env, "get_info"):
            info = env.get_info()
        return obs, info

    @staticmethod
    def _as_done(value):
        if isinstance(value, torch.Tensor):
            return bool(value.any().item())
        if isinstance(value, np.ndarray):
            return bool(value.any())
        return bool(value)


def create_eval_run_dir(config):
    """Create the shared directory for metrics and optional videos."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root = Path(config.get("eval_result_dir") or "eval_result")
    checkpoint_path = config.get("checkpoint_path")
    model_name = config.get("model_name")
    if not model_name and checkpoint_path:
        model_name = Path(checkpoint_path).name
    save_dir = (
        save_root
        / str(config.get("task_name"))
        / str(config.get("policy_name"))
        / str(config.get("setting"))
        / str(config.get("train_config_name"))
        / str(model_name)
        / timestamp
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def create_video_recorder(config, run_dir=None):
    if not config.get("eval_video_log", False):
        return None

    run_dir = Path(run_dir) if run_dir else create_eval_run_dir(config)
    obs_keys = config.get("eval_video_obs_keys", config.get("eval_video_obs_key", "cam_high"))
    return EpisodeVideoRecorder(
        save_dir=run_dir / "videos",
        obs_keys=obs_keys,
        fps=10,
        crf=23,
    )


def make_env_from_configs(config, gym_config_dict, action_config_dict):
    """Build the gym env using EmbodiChain's config parser.

    This keeps nested robot/sensor/object dictionaries as EmbodiChain config
    objects instead of passing raw dicts into EmbodiedEnvCfg.
    """
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    gym_config = deepcopy(gym_config_dict)
    _set_default(gym_config, "num_envs", int(config.get("num_envs", 1)))
    _set_default(gym_config, "device", config.get("device", "cpu"))
    _set_default(gym_config, "headless", bool(config.get("headless", False)))
    _set_default(gym_config, "renderer", config.get("renderer", "hybrid"))
    _set_default(gym_config, "gpu_id", int(config.get("gpu_id", 0)))
    _set_default(gym_config, "arena_space", float(config.get("arena_space", 5.0)))

    max_env_steps, _, _ = resolve_episode_max_steps(config, gym_config)
    gym_config["max_episode_steps"] = max_env_steps

    env_cfg = gym_utils.config_to_cfg(
        gym_config,
        manager_modules=CHALLENGE_MANAGER_MODULES,
    )
    env_cfg.filter_dataset_saving = bool(config.get("filter_dataset_saving", True))
    env_cfg.sim_cfg = SimulationManagerCfg(
        headless=gym_config["headless"],
        sim_device=gym_config["device"],
        render_cfg=RenderCfg(renderer=gym_config["renderer"]),
        gpu_id=gym_config["gpu_id"],
        arena_space=gym_config["arena_space"],
    )
    physics_config = gym_config.get("physics", {})
    if "enable_ccd" in physics_config:
        env_cfg.sim_cfg.physics_config.enable_ccd = bool(
            physics_config["enable_ccd"]
        )

    action_kwargs = {}
    if action_config_dict:
        action_kwargs = (
            action_config_dict
            if "action_config" in action_config_dict
            else {"action_config": action_config_dict}
        )

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_kwargs)
    return env, gym_config

def find_gym_config(config):
    """Load configs/<task_name>/<setting>/gym_config.json."""
    task_name = config.get("task_name")
    setting = config.get("setting")
    if not task_name or not setting:
        print("Error: task_name and setting are required to load gym_config")
        sys.exit(1)

    with open(f"configs/{task_name}/{setting}/gym_config.json", "r") as f:
        return json.load(f)

def find_action_config(config):
    """Load the task action_config.json, falling back to the setting folder."""
    task_name = config.get("task_name")
    setting = config.get("setting")
    if not task_name:
        print("Error: task_name is required to load action_config")
        sys.exit(1)

    action_cfg_path = f"configs/{task_name}/action_config.json"
    if not os.path.exists(action_cfg_path) and setting:
        action_cfg_path = f"configs/{task_name}/{setting}/action_config.json"

    with open(action_cfg_path, "r") as f:
        return json.load(f)

def _instruction_to_text(instruction):
    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, dict):
        for key in ("lang", "text", "prompt"):
            value = instruction.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _find_instruction_recursive(value):
    if isinstance(value, dict):
        text = _instruction_to_text(value.get("instruction"))
        if text:
            return text
        for child in value.values():
            text = _find_instruction_recursive(child)
            if text:
                return text
    elif isinstance(value, list):
        for child in value:
            text = _find_instruction_recursive(child)
            if text:
                return text
    return None


def extract_instruction_from_gym_config(gym_config):
    """Return the task language instruction declared in gym_config.json."""
    dataset_cfg = gym_config.get("env", {}).get("dataset", {})
    if isinstance(dataset_cfg, dict):
        for functor_cfg in dataset_cfg.values():
            if not isinstance(functor_cfg, dict):
                continue
            params = functor_cfg.get("params", {})
            text = _instruction_to_text(params.get("instruction"))
            if text:
                return text

    return _find_instruction_recursive(gym_config)


def _read_cpu_model():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def collect_platform_metadata():
    """Describe the hardware/software platform that gives timing its context."""
    accelerators = []
    if torch.cuda.is_available():
        accelerators = [
            torch.cuda.get_device_name(device_index)
            for device_index in range(torch.cuda.device_count())
        ]
    return {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "cpu": _read_cpu_model(),
        "accelerators": accelerators,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def format_platform(platform_metadata):
    accelerators = platform_metadata["accelerators"]
    accelerator_text = ", ".join(accelerators) if accelerators else "none detected"
    return (
        f"GPU={accelerator_text}; CPU={platform_metadata['cpu']}; "
        f"OS={platform_metadata['operating_system']}; "
        f"PyTorch={platform_metadata['torch_version']}; "
        f"CUDA={platform_metadata['torch_cuda_version']}"
    )


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="RoboSynChallenge Unified Policy Evaluator")

    # Required: config file
    parser.add_argument("--config", type=str, required=True,
                        help="Path to policy YAML config (e.g. policy/pi0/deploy_policy.yml)")

    # Override any config value from command line
    parser.add_argument("--overrides", nargs=argparse.REMAINDER,
                        help="Override config values, e.g. --train_config_name my_config")

    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    if args.overrides:
        for i in range(0, len(args.overrides), 2):
            key = args.overrides[i].lstrip("-")
            value = args.overrides[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            config[key] = value

    return config


def main():
    config = parse_args_and_config()

    policy_name = config.get("policy_name")
    if not policy_name:
        print("Error: 'policy_name' is required in config")
        sys.exit(1)

    task_name = config.get("task_name")
    max_episodes = config.get("max_episodes")
    seed = config.get("seed")
    fixed_episode_seed = config.get("eval_fixed_episode_seed")
    headless = config.get("headless")

    # Load policy adapter
    print(f"Loading policy: {policy_name}")
    policy_pkg = load_policy_adapter(policy_name)

    # Resolve gym and action configs
    gym_config_dict = find_gym_config(config)
    action_config_dict = find_action_config(config)
    max_env_steps, deploy_max_steps, gym_max_steps = resolve_episode_max_steps(
        config, gym_config_dict
    )

    # Build environment
    env_id = gym_config_dict.get("id")
    print(f"Creating environment: {env_id}")
    env, gym_config_dict = make_env_from_configs(config, gym_config_dict, action_config_dict)
    instruction = extract_instruction_from_gym_config(gym_config_dict)
    if instruction:
        env._current_instruction = instruction
        print(f"Using instruction from gym_config: {instruction}")
    run_dir = create_eval_run_dir(config)
    result_path = run_dir / "evaluation_metrics.json"
    video_recorder = create_video_recorder(config, run_dir=run_dir)
    reset_sync_steps = int(config.get("eval_reset_sync_steps", 0))
    eval_env = (
        RecordingEnvProxy(env, video_recorder, reset_sync_steps=reset_sync_steps)
        if video_recorder or reset_sync_steps > 0
        else env
    )
    if video_recorder:
        print(f"Recording eval videos to: {video_recorder.save_dir}")

    # Create model
    print("Creating model...")
    model = policy_pkg.get_model(config)
    print("Model created successfully.")

    # Evaluation loop
    platform_metadata = collect_platform_metadata()
    rng = np.random.RandomState(seed)
    success_count = 0
    action_steps = []
    all_inference_times = []
    episode_inference_totals = []
    print(f"\n{'='*25} Starting Evaluation {'='*25}\n")
    print(f"  Policy: {policy_name}  |  Task: {task_name}")
    print(f"  Episodes: {max_episodes}  |  Seed: {seed}")
    print(
        f"  Max env steps: {max_env_steps} "
        f"(deploy_config.max_steps={deploy_max_steps}, "
        f"gym_config.max_episode_steps={gym_max_steps})"
    )
    print(f"  Timing platform: {format_platform(platform_metadata)}")
    print(f"  Metrics file: {result_path}")
    print(f"{'='*70}\n")

    try:
        for episode in range(max_episodes):
            ep_seed = (
                int(fixed_episode_seed)
                if fixed_episode_seed is not None
                else int(rng.randint(0, 2**31 - 1))
            )
            if video_recorder:
                video_recorder.start_episode(episode, ep_seed)

            episode_success = False
            inference_times = []
            env_steps = 0
            progress_bar = None
            try:
                obs, info = eval_env.reset(seed=ep_seed)
                policy_pkg.reset_model(model)
                progress_bar = tqdm(
                    total=max_env_steps,
                    desc=f"Episode {episode + 1:03d}/{max_episodes:03d}",
                    unit="step",
                    dynamic_ncols=True,
                    leave=False,
                )

                while env_steps < max_env_steps:
                    obs, info, truncated, new_times = policy_pkg.eval(
                        eval_env, model, obs
                    )
                    inference_times.extend(new_times)
                    env_steps = int(info["elapsed_steps"].item())

                    progress_bar.update(
                        max(0, min(env_steps, max_env_steps) - progress_bar.n)
                    )
                    progress_bar.set_postfix(
                        inference=len(inference_times),
                        infer_time=f"{new_times[-1]:.3f}s" if new_times else "cached",
                    )

                    is_truncated = RecordingEnvProxy._as_done(truncated)
                    if not is_truncated and env.get_wrapper_attr("is_task_success")():
                        episode_success = True
                        progress_bar.write("Task success!")
                        break
                    if is_truncated:
                        progress_bar.write("Task timeout!")
                        break
            finally:
                if progress_bar:
                    progress_bar.close()
                if video_recorder:
                    video_recorder.close_episode(success=episode_success)

            success_count += int(episode_success)
            effective_steps = env_steps if episode_success else max_env_steps
            action_steps.append(effective_steps)
            all_inference_times.extend(inference_times)
            episode_inference_totals.append(sum(inference_times))
            episode_average = float(np.mean(inference_times)) if inference_times else 0
            status = "\033[92mSUCCESS\033[0m" if episode_success else "\033[91mFAIL\033[0m"
            print(
                f"  Episode {episode+1}: {status}; action_steps="
                f"{effective_steps}/{max_env_steps}; inference="
                f"{episode_average:.6f}s "
                f"over {len(inference_times)} calls"
            )
            print(f"  [{episode+1:3d}/{max_episodes}] success rate: "
                  f"{success_count}/{episode+1} = {100*success_count/(episode+1):.1f}%")
    finally:
        if video_recorder:
            video_recorder.close_episode(success=False)
        env.close()

    inference_call_count = len(all_inference_times)
    average_action_steps = float(np.mean(action_steps))
    summary = {
        "episode_count": len(action_steps),
        "success_count": success_count,
        "success_rate": success_count / len(action_steps),
        "average_action_steps": average_action_steps,
        "average_action_steps_ratio": average_action_steps / max_env_steps,
        "inference_call_count": inference_call_count,
        "average_inference_calls_per_episode": inference_call_count / len(action_steps),
        "average_inference_time_seconds": (
            float(np.mean(all_inference_times)) if all_inference_times else None
        ),
        "average_inference_time_per_episode_seconds": float(
            np.mean(episode_inference_totals)
        ),
    }
    result_payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "config": {
            "policy": policy_name,
            "task": task_name,
            "setting": config.get("setting"),
            "checkpoint_path": config.get("checkpoint_path"),
            "episode_count": max_episodes,
            "timeout_action_steps": max_env_steps,
            "seed": seed,
        },
        "inference_timing_scope": (
            "raw observation preprocessing and transfer through executable action; "
            "env.step excluded"
        ),
        "platform": platform_metadata,
        "summary": summary,
    }
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(result_payload, result_file, indent=2, ensure_ascii=False)
        result_file.write("\n")

    average_inference_time = summary["average_inference_time_seconds"]
    print(f"\n{'='*50}")
    print(
        f"  Evaluation Results Summary: {success_count}/{max_episodes} "
        f"({100*success_count/max_episodes:.1f}%)"
    )
    print(
        f"  Average action steps: {summary['average_action_steps']:.2f}/"
        f"{max_env_steps} ({100*summary['average_action_steps_ratio']:.2f}%; "
        f"failed episodes count as {max_env_steps} steps)"
    )
    if average_inference_time is None:
        print("  Average inference time: n/a (no inference calls recorded)")
    else:
        print(
            f"  Average inference latency: {average_inference_time:.6f}s "
            f"({1000*average_inference_time:.3f}ms) over "
            f"{summary['inference_call_count']} model calls"
        )
        print(
            "  Average total inference time per episode: "
            f"{summary['average_inference_time_per_episode_seconds']:.6f}s "
            f"over {summary['average_inference_calls_per_episode']:.2f} model calls"
        )
    print(f"  Timing platform: {format_platform(platform_metadata)}")
    print(f"  Metrics saved to: {result_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
