"""Evaluate an OpenPI pi0 policy server inside an EmbodiChain environment.

This script intentionally keeps OpenPI model inference in the OpenPI websocket
server process. The client only measures and aggregates the server-reported
``policy_timing.infer_ms`` value, which is produced around model.sample_actions()
and therefore excludes websocket, observation conversion, simulator stepping and
postprocessing time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # Allows --help in bare system Python.
    np = None

try:
    import torch
except ModuleNotFoundError:  # Allows --help in bare system Python.
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[2]

for path in (
    REPO_ROOT / "EmbodiChain",
    REPO_ROOT / "RoboSynChallenge",
    REPO_ROOT / "openpi" / "src",
    REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src",
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


@dataclass
class EpisodeResult:
    episode: int
    seed: int | None
    success: bool
    terminated: bool
    truncated: bool
    action_steps: int
    model_infer_calls: int
    model_forward_ms: list[float]
    mean_model_forward_ms: float | None
    total_model_forward_ms: float
    wall_time_s: float
    action_results: list[ActionStepResult] | None = None


@dataclass
class ActionStepResult:
    step: int
    model_infer_call: int
    chunk_index: int
    chunk_length: int
    from_new_inference: bool
    model_forward_ms: float | None
    raw_action: list[float]
    env_action: list[float]
    raw_action_dim: int
    env_action_dim: int
    clipped: bool
    truncated_to_action_dim: bool
    success: bool
    terminated: bool
    truncated: bool


@dataclass
class PolicyActionResult:
    action: Any
    model_infer_call: int
    chunk_index: int
    chunk_length: int
    from_new_inference: bool
    model_forward_ms: float | None


class OpenPIChunkPolicy:
    """Caches OpenPI action chunks and records timing only once per model call."""

    def __init__(
        self,
        host: str,
        port: int | None,
        api_key: str | None = None,
        *,
        ws_ping_interval: float | None = None,
        ws_ping_timeout: float | None = None,
        ws_close_timeout: float | None = 1.0,
    ):
        self._policy = _EvalWebsocketClientPolicy(
            host=host,
            port=port,
            api_key=api_key,
            ping_interval=ws_ping_interval,
            ping_timeout=ws_ping_timeout,
            close_timeout=ws_close_timeout,
        )
        self._chunk: np.ndarray | None = None
        self._chunk_index = 0
        self.infer_calls = 0
        self.model_forward_ms: list[float] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.get_server_metadata()

    def reset_episode_stats(self) -> None:
        self._chunk = None
        self._chunk_index = 0
        self.infer_calls = 0
        self.model_forward_ms.clear()

    def next_action(self, obs: dict[str, Any]) -> PolicyActionResult:
        from_new_inference = False
        model_forward_ms = None
        if self._chunk is None or self._chunk_index >= len(self._chunk):
            result = self._policy.infer(obs)
            if "actions" not in result:
                raise KeyError(f"OpenPI response has no 'actions' key: {result.keys()}")
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2:
                raise ValueError(f"Expected action chunk [T, D] or [D], got shape {actions.shape}")

            self._chunk = actions
            self._chunk_index = 0
            self.infer_calls += 1
            from_new_inference = True

            timing = result.get("policy_timing", result.get("server_timing", {}))
            if "infer_ms" in timing:
                model_forward_ms = float(timing["infer_ms"])
                self.model_forward_ms.append(model_forward_ms)

        chunk_index = self._chunk_index
        action = self._chunk[chunk_index]
        self._chunk_index += 1
        return PolicyActionResult(
            action=action,
            model_infer_call=self.infer_calls,
            chunk_index=chunk_index,
            chunk_length=len(self._chunk),
            from_new_inference=from_new_inference,
            model_forward_ms=model_forward_ms,
        )

    def close(self) -> None:
        self._policy.close()


class _EvalWebsocketClientPolicy:
    """OpenPI websocket client with eval-friendly keepalive settings.

    The upstream OpenPI client uses the websockets defaults. During the first
    JAX compile or a long policy call, those defaults can close the connection
    before the server has a chance to respond. Keeping this small client local
    avoids modifying OpenPI itself.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        *,
        ping_interval: float | None = None,
        ping_timeout: float | None = None,
        close_timeout: float | None = 1.0,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        from openpi_client import msgpack_numpy

        self._msgpack_numpy = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._close_timeout = close_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(self):
        import websockets.sync.client

        print(f"Waiting for OpenPI server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    close_timeout=self._close_timeout,
                )
                metadata = self._msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                print("Still waiting for OpenPI server...")
                time.sleep(5)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._msgpack_numpy.unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def _add_env_args(parser: argparse.ArgumentParser) -> None:
    from embodichain.lab.gym.utils.gym_utils import (
        add_env_launcher_args_to_parser,
    )

    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    parser.set_defaults(viser_image_fps=None)
    parser.add_argument(
        "--enable_rt",
        action="store_true",
        help="Deprecated alias for --renderer fast-rt.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenPI pi0 websocket policy in EmbodiChain."
    )
    _add_env_args(parser)

    parser.add_argument("--host", default="127.0.0.1", help="OpenPI policy server host.")
    parser.add_argument("--port", default=8000, type=int, help="OpenPI policy server port.")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--episodes", default=10, type=int, help="Number of evaluation episodes.")
    parser.add_argument("--max_steps", default=None, type=int, help="Override max env steps per episode.")
    parser.add_argument("--seed", default=None, type=int, help="Base reset seed. Episode i uses seed+i.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Language instruction sent to OpenPI. If omitted, it is read from gym config when possible.",
    )
    parser.add_argument(
        "--policy_input_format",
        default="embodichain",
        choices=("embodichain", "aloha"),
        help="Observation key layout expected by the OpenPI training config.",
    )
    parser.add_argument("--state_key", default="robot/qpos", help="Slash path for policy state in env obs.")
    parser.add_argument("--image_key", default="color", help="Camera data key in EmbodiChain sensor obs.")
    parser.add_argument("--cam_high", default="cam_high")
    parser.add_argument("--cam_left_wrist", default="cam_left_wrist")
    parser.add_argument("--cam_right_wrist", default="cam_right_wrist")
    parser.set_defaults(clip_actions=True)
    clip_group = parser.add_mutually_exclusive_group()
    clip_group.add_argument(
        "--clip_actions",
        dest="clip_actions",
        action="store_true",
        help="Clip OpenPI actions to env.single_action_space bounds before env.step().",
    )
    clip_group.add_argument(
        "--no_clip_actions",
        dest="clip_actions",
        action="store_false",
        help="Disable action clipping before env.step().",
    )
    parser.add_argument(
        "--output",
        default="openpi0_embodichain_eval.json",
        help="Path to write detailed JSON results.",
    )
    parser.add_argument(
        "--record_action_results",
        "--record_actions",
        dest="record_action_results",
        default=False,
        action="store_true",
        help="Store every policy action and executed env action in the output JSON.",
    )
    args = parser.parse_args()
    if args.enable_rt and args.renderer is None:
        args.renderer = "fast-rt"
    return args


def _import_embodichain_runtime() -> None:
    import robosynchallenge  # noqa: F401
    import embodichain.lab.gym.utils.gym_utils as gym_utils

    gym_utils.DEFAULT_MANAGER_MODULES = gym_utils.DEFAULT_MANAGER_MODULES + [
        "robosynchallenge.managers.actions",
        "robosynchallenge.managers.datasets",
        "robosynchallenge.managers.events",
        "robosynchallenge.managers.observations",
    ]


def make_env(args: argparse.Namespace):
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    if args.max_steps is not None:
        env_cfg.max_episode_steps = args.max_steps
        gym_config["max_episode_steps"] = args.max_steps

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    return env, gym_config


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _get_by_slash_path(data: Any, path: str) -> Any:
    cur = data
    for key in path.split("/"):
        if isinstance(cur, dict):
            cur = cur[key]
        else:
            cur = cur[key]
    return cur


def _select_env(array: np.ndarray, env_index: int = 0) -> np.ndarray:
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def _extract_state(obs: Any, state_key: str, env_index: int, action_dim: int) -> np.ndarray:
    state = _to_numpy(_get_by_slash_path(obs, state_key))
    state = _select_env(state, env_index=env_index).astype(np.float32, copy=False).reshape(-1)
    if state.shape[0] < action_dim:
        raise ValueError(
            f"State from '{state_key}' has dim {state.shape[0]}, smaller than action dim {action_dim}."
        )
    if state.shape[0] > action_dim:
        state = state[:action_dim]
    return state


def _extract_camera(
    obs: Any,
    sensor_name: str,
    image_key: str,
    env_index: int,
    *,
    channel_first: bool,
) -> np.ndarray:
    sensor_obs = _get_by_slash_path(obs, f"sensor/{sensor_name}/{image_key}")
    image = _to_numpy(sensor_obs)
    image = _select_env(image, env_index=env_index)

    if image.ndim != 3:
        raise ValueError(f"Camera '{sensor_name}/{image_key}' should be 3D, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        chw = image
    else:
        chw = np.moveaxis(image, -1, 0)

    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)
    elif chw.shape[0] > 3:
        chw = chw[:3]

    if np.issubdtype(chw.dtype, np.floating):
        max_value = float(np.nanmax(chw)) if chw.size else 1.0
        if max_value <= 1.0:
            chw = chw * 255.0
        chw = np.clip(chw, 0, 255).astype(np.uint8)
    elif chw.dtype != np.uint8:
        chw = np.clip(chw, 0, 255).astype(np.uint8)

    if channel_first:
        return np.ascontiguousarray(chw)
    return np.ascontiguousarray(np.moveaxis(chw, 0, -1))


def make_openpi_observation(
    obs: Any,
    *,
    args: argparse.Namespace,
    prompt: str,
    action_dim: int,
    env_index: int = 0,
) -> dict[str, Any]:
    state = _extract_state(obs, args.state_key, env_index, action_dim)
    if args.policy_input_format == "aloha":
        return {
            "state": state,
            "images": {
                "cam_high": _extract_camera(
                    obs, args.cam_high, args.image_key, env_index, channel_first=True
                ),
                "cam_left_wrist": _extract_camera(
                    obs, args.cam_left_wrist, args.image_key, env_index, channel_first=True
                ),
                "cam_right_wrist": _extract_camera(
                    obs, args.cam_right_wrist, args.image_key, env_index, channel_first=True
                ),
            },
            "prompt": prompt,
        }

    return {
        "observation/state": state,
        "observation/image": _extract_camera(
            obs, args.cam_high, args.image_key, env_index, channel_first=False
        ),
        "observation/left_wrist_image": _extract_camera(
            obs, args.cam_left_wrist, args.image_key, env_index, channel_first=False
        ),
        "observation/right_wrist_image": _extract_camera(
            obs, args.cam_right_wrist, args.image_key, env_index, channel_first=False
        ),
        "prompt": prompt,
    }


def _bool_from_tensor(value: Any) -> bool:
    arr = _to_numpy(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(arr.reshape(-1)[0])


def _infer_prompt(gym_config: dict[str, Any]) -> str:
    try:
        return gym_config["env"]["dataset"]["lerobot"]["params"]["instruction"]["lang"]
    except KeyError:
        return "perform the task"


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "p50": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if np is None or torch is None:
        raise RuntimeError(
            "This evaluator needs numpy and torch. Run it inside the same Python "
            "environment you use for EmbodiChain/OpenPI, not the bare system Python."
        )

    _import_embodichain_runtime()
    env, gym_config = make_env(args)

    if args.num_envs != 1:
        env.close()
        raise ValueError("OpenPI websocket policies are single-observation policies here; use --num_envs 1.")

    prompt = args.prompt or _infer_prompt(gym_config)
    policy = OpenPIChunkPolicy(args.host, args.port, args.api_key)
    print(f"Connected to OpenPI server. Metadata: {policy.metadata}")
    print(f"Prompt: {prompt!r}")

    action_space = env.unwrapped.single_action_space
    action_dim = int(np.prod(action_space.shape))
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    max_steps = args.max_steps or int(getattr(env.unwrapped, "max_episode_steps", gym_config.get("max_episode_steps", 500)))

    episode_results: list[EpisodeResult] = []

    try:
        for episode in range(args.episodes):
            episode_seed = None if args.seed is None else args.seed + episode
            reset_kwargs = {}
            if episode_seed is not None:
                reset_kwargs["seed"] = episode_seed
            obs, _ = env.reset(**reset_kwargs)
            policy.reset_episode_stats()

            terminated = False
            truncated = False
            success = False
            action_steps = 0
            episode_action_results: list[ActionStepResult] = []
            wall_start = time.monotonic()

            for step in range(max_steps):
                openpi_obs = make_openpi_observation(
                    obs,
                    args=args,
                    prompt=prompt,
                    action_dim=action_dim,
                )
                policy_action = policy.next_action(openpi_obs)
                raw_action = np.asarray(policy_action.action, dtype=np.float32).reshape(-1)
                raw_action_dim = int(raw_action.shape[0])
                if raw_action.shape[0] < action_dim:
                    raise ValueError(
                        f"OpenPI action dim {raw_action.shape[0]} is smaller than env action dim {action_dim}."
                    )
                truncated_to_action_dim = raw_action.shape[0] > action_dim
                action = raw_action[:action_dim] if truncated_to_action_dim else raw_action

                clipped = False
                if args.clip_actions:
                    clipped_action = np.clip(action, low, high)
                    clipped = not np.allclose(
                        action,
                        clipped_action,
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                    action = clipped_action

                action_tensor = torch.as_tensor(
                    action[None, :],
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                obs, _, terminated_t, truncated_t, info = env.step(action_tensor)
                action_steps += 1

                success = _bool_from_tensor(info.get("success", terminated_t))
                terminated = _bool_from_tensor(terminated_t)
                truncated = _bool_from_tensor(truncated_t)
                if args.record_action_results:
                    episode_action_results.append(
                        ActionStepResult(
                            step=step,
                            model_infer_call=policy_action.model_infer_call,
                            chunk_index=policy_action.chunk_index,
                            chunk_length=policy_action.chunk_length,
                            from_new_inference=policy_action.from_new_inference,
                            model_forward_ms=policy_action.model_forward_ms,
                            raw_action=raw_action.tolist(),
                            env_action=action.astype(np.float32, copy=False).tolist(),
                            raw_action_dim=raw_action_dim,
                            env_action_dim=action_dim,
                            clipped=clipped,
                            truncated_to_action_dim=truncated_to_action_dim,
                            success=success,
                            terminated=terminated,
                            truncated=truncated,
                        )
                    )
                if terminated or truncated:
                    break

            infer_times = list(policy.model_forward_ms)
            result = EpisodeResult(
                episode=episode,
                seed=episode_seed,
                success=success,
                terminated=terminated,
                truncated=truncated,
                action_steps=action_steps,
                model_infer_calls=policy.infer_calls,
                model_forward_ms=infer_times,
                mean_model_forward_ms=float(np.mean(infer_times)) if infer_times else None,
                total_model_forward_ms=float(np.sum(infer_times)) if infer_times else 0.0,
                wall_time_s=time.monotonic() - wall_start,
                action_results=episode_action_results if args.record_action_results else None,
            )
            episode_results.append(result)
            print(
                "episode={episode} success={success} action_steps={steps} "
                "model_calls={calls} mean_forward_ms={mean}".format(
                    episode=episode,
                    success=result.success,
                    steps=result.action_steps,
                    calls=result.model_infer_calls,
                    mean=(
                        "n/a"
                        if result.mean_model_forward_ms is None
                        else f"{result.mean_model_forward_ms:.2f}"
                    ),
                )
            )
    finally:
        env.close()

    successes = [r.success for r in episode_results]
    all_forward_ms = [value for result in episode_results for value in result.model_forward_ms]
    summary = {
        "episodes": len(episode_results),
        "successes": int(sum(successes)),
        "success_rate": float(sum(successes) / len(successes)) if successes else math.nan,
        "action_steps": _stats([float(r.action_steps) for r in episode_results]),
        "model_infer_calls": _stats([float(r.model_infer_calls) for r in episode_results]),
        "episode_mean_model_forward_ms": _stats(all_forward_ms),
        "total_model_forward_ms": float(sum(r.total_model_forward_ms for r in episode_results)),
    }

    payload = {
        "task": gym_config.get("id"),
        "gym_config": args.gym_config,
        "action_config": args.action_config,
        "prompt": prompt,
        "policy_input_format": args.policy_input_format,
        "clip_actions": args.clip_actions,
        "record_action_results": args.record_action_results,
        "openpi_server_metadata": policy.metadata,
        "summary": summary,
        "episodes": [asdict(result) for result in episode_results],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote results to {output_path}")
    return payload


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
