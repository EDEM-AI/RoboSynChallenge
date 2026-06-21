#!/usr/bin/env python
# ----------------------------------------------------------------------------
# RoboSynChallenge — 统一策略评估脚本
#
# 设计模式参考 RoboTwin: 通过 --policy_name 参数动态加载不同策略适配层
# 每个策略在 policy/<policy_name>/deploy_policy.py 中提供:
#   get_model(usr_args)   — 创建策略模型实例
#   eval(env, model, obs)  — 执行一次推理并作用于环境
#   reset_model(model)     — 重置策略内部状态
#
# 用法:
#   python scripts/eval_policy.py --config policy/pi0/deploy_policy.yml \
#       --policy_name pi0 --task_name click_bell --setting random \
#       --train_config_name ... --model_name ... --checkpoint_id 30000 \
#       --max_episodes 100 --seed 0
# ----------------------------------------------------------------------------

import os
import sys
import argparse
import importlib
import json
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import yaml

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
        eval(env: gym.Env, model, obs) -> obs
            Optionally return (obs, diagnostics: dict). Diagnostics can include
            "state" and "actions"; timing is measured by this evaluator.
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


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


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


def get_policy_eval_step_size(policy_name, config):
    if policy_name == "pi0":
        return int(config.get("pi0_step", 50))
    return int(config.get("policy_eval_step_size", 1))


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _any_bool(value):
    if value is None:
        return False
    return bool(np.asarray(_to_numpy(value)).any())


def _format_array_for_log(value):
    return np.array2string(
        np.asarray(value),
        precision=6,
        separator=", ",
        suppress_small=False,
        threshold=sys.maxsize,
        max_line_width=200,
    )


def _split_policy_eval_result(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    return result, {}


def _print_policy_eval_info(
    policy_name,
    episode,
    max_episodes,
    infer_call,
    env_steps_after,
    max_env_steps,
    env_steps_this_infer,
    policy_eval_time_s,
    diagnostics,
):
    print(f"\n{'='*50}")
    print(f"Episode {episode:03d}/{max_episodes:03d} | "
          f"{policy_name} INFER {infer_call:04d} | "
          f"Env Step {env_steps_after:04d}/{max_env_steps:04d}")
    print(f"{'='*50}\n")

    if diagnostics.get("image_available") is not None:
        print(f"image_available={diagnostics['image_available']}")
    if diagnostics.get("image_shapes") is not None:
        print(f"image_shapes={diagnostics['image_shapes']}")
    if diagnostics.get("state_shape") is not None:
        print(f"state_shape={diagnostics['state_shape']}")
    if diagnostics.get("state") is not None:
        print(f"input_state={_format_array_for_log(diagnostics['state'])}")
    if diagnostics.get("actions") is not None:
        actions = np.asarray(diagnostics["actions"])
        print(
            f"actions_shape={actions.shape}, "
            f"actions={_format_array_for_log(actions)}"
        )
    print(f"env_steps_this_infer={env_steps_this_infer}")
    print(f"policy_eval_time_s={policy_eval_time_s:.6f}")


def _get_embodichain_camera_color(obs, camera_uid):
    try:
        return obs["sensor"][camera_uid]["color"]
    except Exception:
        return None


def _format_video_frame(frame):
    frame = _to_numpy(frame)

    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"Expected video frame shape (H, W, C), got {frame.shape}.")

    if frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame with 3 channels, got {frame.shape}.")

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and np.max(frame) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frame)


class EpisodeVideoRecorder:
    """Record one RGB observation stream per episode through an ffmpeg pipe."""

    def __init__(self, save_dir, obs_key="cam_high", fps=10, crf=23):
        self.save_dir = Path(save_dir)
        self.obs_key = obs_key
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

        frame = _get_embodichain_camera_color(obs, self.obs_key)
        if frame is None:
            raise KeyError(
                f"EmbodiChain camera color observation 'sensor/{self.obs_key}/color' "
                "not found for video recording."
            )

        frame = _format_video_frame(frame)
        height, width = frame.shape[:2]
        if self._process is None:
            self._start_writer(width, height)
        elif self._frame_size != (width, height):
            raise ValueError(
                f"Video frame size changed from {self._frame_size} to {(width, height)}."
            )

        self._process.stdin.write(frame.tobytes())
        self._frame_count += 1

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

    def __init__(self, env, recorder):
        self._env = env
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        self._recorder.record(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._recorder.record(obs)
        return obs, reward, terminated, truncated, info


def create_video_recorder(config):
    if not _as_bool(config.get("eval_video_log", False)):
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root = Path(config.get("eval_save_dir", "eval_result"))
    save_dir = (
        save_root
        / str(config.get("task_name", "unknown_task"))
        / str(config.get("policy_name", "unknown_policy"))
        / str(config.get("setting", "unknown_setting"))
        / str(config.get("train_config_name", "unknown_config"))
        / str(config.get("model_name", "unknown_model"))
        / str(config.get("checkpoint_id", "unknown_ckpt"))
        / timestamp
        / "videos"
    )
    return EpisodeVideoRecorder(
        save_dir=save_dir,
        obs_key=config.get("eval_video_obs_key", "cam_high"),
        fps=config.get("eval_video_fps", 10),
        crf=config.get("eval_video_crf", 23),
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

    action_kwargs = {}
    if action_config_dict:
        action_kwargs = (
            action_config_dict
            if "action_config" in action_config_dict
            else {"action_config": action_config_dict}
        )

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_kwargs)
    return env, gym_config


def load_json_config(path):
    with open(path, "r") as f:
        return json.load(f)


def find_gym_config(config):
    """Load configs/<task_name>/<setting>/gym_config.json."""
    task_name = config.get("task_name")
    setting = config.get("setting")
    if not task_name or not setting:
        print("Error: task_name and setting are required to load gym_config")
        sys.exit(1)

    return load_json_config(f"configs/{task_name}/{setting}/gym_config.json")


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

    return load_json_config(action_cfg_path)


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
    policy_eval_step_size = get_policy_eval_step_size(policy_name, config)

    # Build environment
    env_id = gym_config_dict.get("id")
    print(f"Creating environment: {env_id}")
    env, gym_config_dict = make_env_from_configs(config, gym_config_dict, action_config_dict)
    instruction = extract_instruction_from_gym_config(gym_config_dict)
    if instruction:
        env._current_instruction = instruction
        print(f"Using instruction from gym_config: {instruction}")
    video_recorder = create_video_recorder(config)
    eval_env = RecordingEnvProxy(env, video_recorder) if video_recorder else env
    if video_recorder:
        print(f"Recording eval videos to: {video_recorder.save_dir}")

    # Create model
    print("Creating model...")
    model = policy_pkg.get_model(config)
    print("Model created successfully.")

    # Evaluation loop
    rng = np.random.RandomState(seed)
    success_count = 0
    print(f"\n{'='*50}")
    print(f"  Policy: {policy_name}  |  Task: {task_name}")
    print(f"  Episodes: {max_episodes}  |  Seed: {seed}")
    print(
        f"  Max env steps: {max_env_steps} "
        f"(deploy_config.max_steps={deploy_max_steps}, "
        f"gym_config.max_episode_steps={gym_max_steps})"
    )
    print(f"  Policy eval step size: {policy_eval_step_size}")
    print(f"{'='*50}\n")

    try:
        for episode in range(max_episodes):
            ep_seed = int(rng.randint(0, 2**31 - 1))
            if video_recorder:
                video_recorder.start_episode(episode, ep_seed)

            episode_success = False
            episode_policy_eval_times = []
            env_steps = 0
            inference_count = 0
            try:
                obs, info = eval_env.reset(seed=ep_seed)
                policy_pkg.reset_model(model)

                while env_steps < max_env_steps:
                    remaining_env_steps = max_env_steps - env_steps
                    if remaining_env_steps < policy_eval_step_size:
                        print(
                            f"### EP {episode+1:03d}/{max_episodes:03d} | "
                            f"Tail Dropped | nv Step {env_steps:04d}/{max_env_steps:04d} "
                            f"| remaining={remaining_env_steps} < "
                            f"policy_eval_step_size={policy_eval_step_size} ###"
                        )
                        break
                    inference_count += 1
                    policy_eval_start = time.perf_counter()
                    eval_result = policy_pkg.eval(eval_env, model, obs)
                    policy_eval_time_s = time.perf_counter() - policy_eval_start
                    episode_policy_eval_times.append(policy_eval_time_s)
                    obs, diagnostics = _split_policy_eval_result(eval_result)
                    env_steps_this_infer = int(diagnostics.get("env_steps_executed", 1))
                    if env_steps_this_infer <= 0:
                        raise RuntimeError(
                            f"Policy '{policy_name}' reported env_steps_executed="
                            f"{env_steps_this_infer}; cannot advance episode."
                        )
                    env_steps += env_steps_this_infer
                    _print_policy_eval_info(
                        policy_name,
                        episode + 1,
                        max_episodes,
                        inference_count,
                        env_steps,
                        max_env_steps,
                        env_steps_this_infer,
                        policy_eval_time_s,
                        diagnostics,
                    )

                    success = getattr(eval_env, "is_task_success", lambda: torch.tensor(False))()
                    if _any_bool(success):
                        episode_success = True
                        break
                    if _any_bool(diagnostics.get("terminated")) or _any_bool(
                        diagnostics.get("truncated")
                    ):
                        break
            finally:
                if video_recorder:
                    video_recorder.close_episode(success=episode_success)

            if episode_policy_eval_times:
                avg_policy_eval_time = float(np.mean(episode_policy_eval_times))
                print(
                    f"  Episode {episode+1} policy eval timing: "
                    f"avg={avg_policy_eval_time:.6f}s over "
                    f"{len(episode_policy_eval_times)} inference calls, "
                    f"env_steps={env_steps}/{max_env_steps}"
                )
            else:
                print(f"  Episode {episode+1} policy eval timing: no eval steps recorded")

            if episode_success:
                success_count += 1
                status = "\033[92mSUCCESS\033[0m"
            else:
                status = "\033[91mFAIL\033[0m"

            print(f"  [{episode+1:3d}/{max_episodes}] {status}  "
                  f"(success rate: {success_count}/{episode+1} = {100*success_count/(episode+1):.1f}%)")
    finally:
        if video_recorder:
            video_recorder.close_episode(success=False)
        env.close()

    print(f"\n{'='*50}")
    print(f"  Final: {success_count}/{max_episodes} ({100*success_count/max_episodes:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
