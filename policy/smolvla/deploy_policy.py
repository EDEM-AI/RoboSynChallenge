# ----------------------------------------------------------------------------
# SmolVLA Policy Adapter for RoboSynChallenge
# ----------------------------------------------------------------------------

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _select_env(value: np.ndarray, env_index: int = 0) -> np.ndarray:
    if value.ndim == 0:
        return value
    return value[env_index]


def _extract_image(obs, sensor_name: str, env_index: int = 0) -> np.ndarray:
    image = _to_numpy(obs["sensor"][sensor_name]["color"])
    image = _select_env(image, env_index=env_index)
    if image.ndim != 3:
        raise ValueError(f"Expected image for {sensor_name} to be 3D, got {image.shape}")

    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        chw = image
    else:
        chw = np.moveaxis(image, -1, 0)

    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)
    elif chw.shape[0] > 3:
        chw = chw[:3]

    if np.issubdtype(chw.dtype, np.floating):
        if float(np.nanmax(chw)) <= 1.0:
            chw = chw * 255.0
        chw = np.clip(chw, 0, 255).astype(np.uint8)
    elif chw.dtype != np.uint8:
        chw = np.clip(chw, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(chw)


def _extract_joint(obs, key: str, fallback: np.ndarray | None = None, env_index: int = 0) -> np.ndarray:
    robot_obs = obs["robot"]
    if key in robot_obs:
        value = _to_numpy(robot_obs[key])
        value = _select_env(value, env_index=env_index).astype(np.float32, copy=False).reshape(-1)
        return np.ascontiguousarray(value)
    if fallback is None:
        raise KeyError(f"Missing robot observation key: {key}")
    return np.zeros_like(fallback, dtype=np.float32)


def encode_obs(obs, image_keys: dict[str, str] | None = None):
    state = _extract_joint(obs, "qpos")
    qvel = _extract_joint(obs, "qvel", fallback=state)
    qf = _extract_joint(obs, "qf", fallback=state)

    encoded = {
        "observation.state": state,
        "observation.qvel": qvel,
        "observation.qf": qf,
    }
    image_keys = image_keys or {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_left_wrist": "cam_left_wrist",
        "observation.images.cam_right_wrist": "cam_right_wrist",
    }
    for obs_key, sensor_name in image_keys.items():
        encoded[obs_key] = _extract_image(obs, sensor_name)
    return encoded


def _resolve_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"auto", "none", ""}:
            return None
        return normalized in {"1", "true", "yes", "y", "on"}
    return bool(value)


def encode_action(
    action,
    env,
    rescale_gripper: bool = True,
    gripper_indices: tuple[int, ...] = (6, 13),
    gripper_source_range: tuple[float, float] = (0.0, 1.0),
):
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.shape[0] < env_action_dim:
        raise ValueError(
            f"Policy action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
        )
    action_array = action_array[:env_action_dim]
    if rescale_gripper:
        action_array = action_array.copy()
        action_space = env.unwrapped.single_action_space
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        src_low, src_high = gripper_source_range
        src_span = max(src_high - src_low, np.finfo(np.float32).eps)
        for gripper_idx in gripper_indices:
            if gripper_idx >= env_action_dim:
                continue
            gripper_01 = np.clip((action_array[gripper_idx] - src_low) / src_span, 0.0, 1.0)
            action_array[gripper_idx] = low[gripper_idx] + gripper_01 * (
                high[gripper_idx] - low[gripper_idx]
            )
    action_tensor = torch.as_tensor(action_array, dtype=torch.float32, device=env.unwrapped.device)
    return action_tensor.unsqueeze(0)


class SmolVLAWorkerClient:
    def __init__(self, usr_args: dict):
        policy_root = Path(__file__).resolve().parent
        self.checkpoint_dir = Path(usr_args["checkpoint_dir"]).resolve()
        lerobot_root = (
            usr_args.get("lerobot_root")
            or os.environ.get("SMOLVLA_LEROBOT_ROOT")
            or os.environ.get("LEROBOT_ROOT")
            or ""
        )
        self.lerobot_root = Path(lerobot_root).expanduser().resolve() if lerobot_root else None
        self.python_bin = usr_args.get("smolvla_python") or os.environ.get("SMOLVLA_PYTHON") or sys.executable
        self.device = usr_args.get("pytorch_device", "cuda")
        self.smolvla_steps = int(usr_args.get("smolvla_steps", 10))
        self.gripper_indices = tuple(int(idx) for idx in usr_args.get("smolvla_gripper_indices", [6, 13]))
        source_range = usr_args.get("smolvla_gripper_source_range", [0.0, 1.0])
        self.gripper_source_range = (float(source_range[0]), float(source_range[1]))
        self.rescale_gripper = self._resolve_gripper_rescale(usr_args.get("smolvla_rescale_gripper", "auto"))
        self.debug_actions = bool(usr_args.get("smolvla_debug_actions", False))
        self.debug_action_chunks = int(usr_args.get("smolvla_debug_action_chunks", 1))
        self.debug_action_log_path = usr_args.get("smolvla_debug_action_log_path")
        self._debug_chunks_printed = 0
        self.cuda_visible_devices = str(
            usr_args.get("smolvla_cuda_visible_devices", usr_args.get("gpu_id", 0))
        )
        self.image_keys = self._resolve_image_keys()
        worker_path = policy_root / "smolvla_worker.py"
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        worker_cmd = [
            self.python_bin,
            str(worker_path),
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--device",
            self.device,
        ]
        if self.lerobot_root is not None:
            worker_cmd.extend(["--lerobot-root", str(self.lerobot_root)])
        self.proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        atexit.register(self.close)

    def _resolve_gripper_rescale(self, override) -> bool:
        override_value = _resolve_optional_bool(override)
        if override_value is not None:
            return override_value

        stats_path = self.checkpoint_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        try:
            from safetensors.torch import load_file

            stats = load_file(str(stats_path))
            action_max = _to_numpy(stats["action.max"]).reshape(-1)
        except Exception:
            # Most RoboSyn CobotMagic tasks use 0-1 gripper data, while the env
            # expects physical qpos. Prefer the safer conversion if stats are unavailable.
            return True

        valid_indices = [idx for idx in self.gripper_indices if idx < action_max.shape[0]]
        gripper_max = action_max[valid_indices]
        return bool(gripper_max.size and float(np.nanmax(gripper_max)) > 0.5)

    def close(self) -> None:
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            return
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def _resolve_image_keys(self) -> dict[str, str]:
        config_path = self.checkpoint_dir / "config.json"
        config = json.loads(config_path.read_text())
        input_features = config.get("input_features", {})

        if "observation.images.camera1" in input_features:
            return {
                "observation.images.camera1": "cam_high",
                "observation.images.camera2": "cam_left_wrist",
                "observation.images.camera3": "cam_right_wrist",
            }

        return {
            "observation.images.cam_high": "cam_high",
            "observation.images.cam_left_wrist": "cam_left_wrist",
            "observation.images.cam_right_wrist": "cam_right_wrist",
        }

    def _rpc(self, payload: dict) -> dict:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("SmolVLA worker pipes are not available.")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("SmolVLA worker exited unexpectedly.")
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "SmolVLA worker error"))
            return response

    def infer(self, obs: dict[str, np.ndarray], task: str) -> list[np.ndarray]:
        fd, obs_path = tempfile.mkstemp(prefix="smolvla_obs_", suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(obs_path, **obs, task=np.asarray(task))
            response = self._rpc(
                {
                    "cmd": "infer",
                    "obs_path": obs_path,
                    "n_action_steps": self.smolvla_steps,
                }
            )
        finally:
            if os.path.exists(obs_path):
                os.remove(obs_path)

        return [np.asarray(action, dtype=np.float32) for action in response["actions"]]

    def reset(self) -> None:
        self._debug_chunks_printed = 0
        self._rpc({"cmd": "reset"})

    def maybe_print_action_debug(self, actions: list[np.ndarray]) -> None:
        if not self.debug_actions or self._debug_chunks_printed >= self.debug_action_chunks:
            return
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim != 2 or action_array.shape[1] <= max(self.gripper_indices, default=-1):
            print(f"[SmolVLA action debug] unexpected action shape: {action_array.shape}")
            self._debug_chunks_printed += 1
            return

        gripper_values = {idx: action_array[:, idx] for idx in self.gripper_indices}
        gripper_summary = " ".join(
            f"gripper[{idx}] first10={np.round(values[:10], 4).tolist()} "
            f"min/max=({values.min():.4f}, {values.max():.4f})"
            for idx, values in gripper_values.items()
        )
        summary = (
            "[SmolVLA action debug] "
            f"chunk={self._debug_chunks_printed} shape={action_array.shape} "
            f"{gripper_summary}"
        )
        detail = (
            "[SmolVLA action debug] "
            f"first action={np.round(action_array[0], 4).tolist()} "
            f"last action={np.round(action_array[-1], 4).tolist()}"
        )
        print(summary, flush=True)
        print(detail, flush=True)
        if self.debug_action_log_path:
            with open(self.debug_action_log_path, "a", encoding="utf-8") as debug_file:
                debug_file.write(summary + "\n")
                debug_file.write(detail + "\n")
        self._debug_chunks_printed += 1

    def maybe_print_env_action_debug(self, action_tensor: torch.Tensor) -> None:
        if not self.debug_actions or self._debug_chunks_printed > self.debug_action_chunks:
            return
        action_array = _to_numpy(action_tensor).reshape(-1)
        if action_array.shape[0] <= max(self.gripper_indices, default=-1):
            return
        gripper_summary = " ".join(
            f"gripper[{idx}]={action_array[idx]:.6f}" for idx in self.gripper_indices
        )
        line = (
            "[SmolVLA env action debug] "
            f"rescale_gripper={self.rescale_gripper} "
            f"source_range={self.gripper_source_range} "
            f"{gripper_summary} "
            f"first env action={np.round(action_array, 6).tolist()}"
        )
        print(line, flush=True)
        if self.debug_action_log_path:
            with open(self.debug_action_log_path, "a", encoding="utf-8") as debug_file:
                debug_file.write(line + "\n")


def get_model(usr_args):
    if "checkpoint_dir" not in usr_args or not usr_args["checkpoint_dir"]:
        raise ValueError("checkpoint_dir must be provided in deploy_policy.yml or overrides.")
    return SmolVLAWorkerClient(usr_args)


def eval(env, model, obs):
    instruction = getattr(env, "_current_instruction", None) or "click the bell"
    encoded_obs = encode_obs(obs, image_keys=model.image_keys)
    actions = model.infer(encoded_obs, task=instruction)
    model.maybe_print_action_debug(actions)

    final_obs = obs
    info = {}
    truncated = False
    for action in actions:
        action_tensor = encode_action(
            action,
            env,
            rescale_gripper=model.rescale_gripper,
            gripper_indices=model.gripper_indices,
            gripper_source_range=model.gripper_source_range,
        )
        model.maybe_print_env_action_debug(action_tensor)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if bool(_to_numpy(truncated).reshape(-1)[0]):
            break

    return final_obs, info, truncated


def reset_model(model):
    model.reset()


def close_model(model):
    model.close()
