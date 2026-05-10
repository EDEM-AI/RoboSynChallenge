"""Estimate a task entity's reset randomization range for expert demos.

The script probes fixed spawn offsets inside the current
pose-randomization ``position_range`` and checks whether the task can generate
an expert action list from the given action config.  It then reports a
recommended axis-aligned range that keeps the sampled success rate high.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # Allows --help and --list-events in bare Python.
    np = None


REPO_ROOT = Path(__file__).resolve().parents[2]

for path in (
    REPO_ROOT / "EmbodiChain",
    REPO_ROOT / "Embodied_Challenge",
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


AXIS_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}
INDEX_TO_AXIS = {0: "x", 1: "y", 2: "z"}


def require_numpy() -> None:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for spawn range analysis. Please run this script "
            "with the project Python environment that has EmbodiChain dependencies."
        )


POSE_RANDOMIZATION_FUNCS = {
    "randomize_rigid_object_pose": "rigid_object",
    "randomize_entity_root_pose_group": "rigid_object",
    "randomize_entity_root_pose_group": "articulation",
    "randomize_articulation_root_pose": "articulation",
}


@dataclass
class PoseRandomizationEvent:
    name: str
    uid: str
    entity_type: str
    func: str
    position_range: list[list[float]]
    relative_position: bool
    relative_rotation: bool
    init_pos: list[float] | None


@dataclass
class PointResult:
    index: int
    point: list[float]
    actual_point: list[float]
    attempts: int
    generation_successes: int
    rollout_successes: int | None
    success_rate: float
    generation_success_rate: float
    errors: list[str]


@dataclass
class TrialResult:
    generated: bool
    rollout_success: bool | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a task entity's random spawn range by probing expert-demo "
            "generation success under a gym_config/action_config pair."
        )
    )
    parser.add_argument("--gym_config", required=True, help="Path to gym_config.json.")
    parser.add_argument(
        "--action_config",
        default=None,
        help="Path to action_config.json. Required unless --list-events is used.",
    )
    parser.add_argument(
        "--uid",
        default=None,
        help="Entity uid in the supported pose randomization event.",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Event name to analyze. Overrides --uid when provided.",
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        help=(
            "List supported pose randomization events in gym_config and exit. "
            "Currently supports randomize_rigid_object_pose and "
            "randomize_articulation_root_pose."
        ),
    )

    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--arena_space", default=5.0, type=float)
    parser.add_argument("--enable_rt", default=False, action="store_true")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--preview", default=False, action="store_true")
    parser.add_argument("--headless", dest="headless", default=True, action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--filter_visual_rand",
        dest="filter_visual_rand",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--no-filter_visual_rand",
        dest="filter_visual_rand",
        action="store_false",
    )
    parser.add_argument(
        "--filter_dataset_saving",
        dest="filter_dataset_saving",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--no-filter_dataset_saving",
        dest="filter_dataset_saving",
        action="store_false",
    )
    parser.add_argument(
        "--filter_distractor_events",
        dest="filter_distractor_events",
        default=True,
        action="store_true",
        help="Filter distractor randomization events before env creation.",
    )
    parser.add_argument(
        "--no-filter_distractor_events",
        dest="filter_distractor_events",
        action="store_false",
        help="Keep distractor randomization events.",
    )

    parser.add_argument(
        "--sample-mode",
        default="grid",
        choices=("grid", "random"),
        help="How to choose fixed candidate positions.",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=None,
        help="Axes to scan, e.g. --axes x y. Defaults to axes with non-zero range.",
    )
    parser.add_argument(
        "--grid-size",
        nargs="+",
        type=int,
        default=[9],
        help="Grid resolution. Use one value for all scanned axes, or one per axis.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of random candidate positions when --sample-mode=random.",
    )
    parser.add_argument(
        "--trials-per-point",
        type=int,
        default=1,
        help="Number of reset/demo-generation attempts at each fixed point.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--point-success-threshold",
        type=float,
        default=1.0,
        help="A probed point is considered safe when its success rate is at least this value.",
    )
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=0.9,
        help="Desired validation success rate for the recommended range.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="Optional random resets used to validate the recommended range. Default is 0.",
    )
    parser.add_argument(
        "--max-shrink-iterations",
        type=int,
        default=5,
        help="Shrink the recommended range toward its center if validation misses the target.",
    )
    parser.add_argument(
        "--shrink-factor",
        type=float,
        default=0.8,
        help="Per-iteration range width multiplier used during validation shrinking.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Extra margin added to recommended scanned-axis bounds before validation.",
    )
    parser.add_argument(
        "--trim-quantile",
        type=float,
        default=0.0,
        help="Trim this quantile from good-point bounds. Useful for noisy random sampling.",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Also execute generated expert actions and require task success.",
    )
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=None,
        help="Maximum number of generated expert actions to execute during --rollout.",
    )
    parser.add_argument(
        "--verbose-failures",
        action="store_true",
        help="Print full tracebacks for failed probes.",
    )

    parser.add_argument("--output", default=None, help="Write JSON report to this path.")
    parser.add_argument("--csv", default=None, help="Write per-point CSV results.")
    parser.add_argument(
        "--plot",
        default=None,
        help=(
            "Path to save the success/failure point visualization. Defaults to "
            "<output>.png when --output is set, otherwise spawn_range_analysis.png."
        ),
    )
    parser.add_argument(
        "--no-plot",
        dest="save_plot",
        action="store_false",
        help="Disable saving the success/failure point visualization.",
    )
    parser.set_defaults(save_plot=True)
    parser.add_argument(
        "--write-gym-config",
        default=None,
        help="Write a copy of gym_config with the recommended range patched in.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def find_entity_init_pos(
    gym_config: dict[str, Any],
    uid: str,
    entity_type: str,
) -> list[float] | None:
    search_keys = {
        "rigid_object": ("rigid_object", "background"),
        "articulation": ("articulation",),
    }.get(entity_type, ("rigid_object", "background", "articulation"))

    for key in search_keys:
        for entity_cfg in gym_config.get(key, []):
            if entity_cfg.get("uid") == uid and "init_pos" in entity_cfg:
                return [float(v) for v in entity_cfg["init_pos"]]

    def walk(value: Any) -> list[float] | None:
        if isinstance(value, dict):
            if value.get("uid") == uid and "init_pos" in value:
                return [float(v) for v in value["init_pos"]]
            for child in value.values():
                ret = walk(child)
                if ret is not None:
                    return ret
        elif isinstance(value, list):
            for child in value:
                ret = walk(child)
                if ret is not None:
                    return ret
        return None

    return walk(gym_config)


def find_entity_type(
    gym_config: dict[str, Any],
    uid: str,
    default: str,
) -> str:
    for entity_type in ("rigid_object", "background", "articulation"):
        for entity_cfg in gym_config.get(entity_type, []):
            if entity_cfg.get("uid") == uid:
                return entity_type
    return default


def iter_pose_randomization_entity_cfgs(params: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(params.get("entity_cfg"), dict):
        return [params["entity_cfg"]]
    return [
        entity_cfg
        for entity_cfg in params.get("entity_cfgs", [])
        if isinstance(entity_cfg, dict)
    ]


def find_pose_randomization_events(
    gym_config: dict[str, Any],
) -> list[PoseRandomizationEvent]:
    events = gym_config.get("env", {}).get("events", {})
    found: list[PoseRandomizationEvent] = []
    for event_name, event_cfg in events.items():
        func = event_cfg.get("func")
        if func not in POSE_RANDOMIZATION_FUNCS:
            continue
        params = event_cfg.get("params", {})
        position_range = params.get("position_range")
        if position_range is None:
            continue
        for entity_cfg in iter_pose_randomization_entity_cfgs(params):
            uid = entity_cfg.get("uid")
            if not uid:
                continue
            entity_type = find_entity_type(
                gym_config, uid, POSE_RANDOMIZATION_FUNCS[func]
            )
            found.append(
                PoseRandomizationEvent(
                    name=event_name,
                    uid=uid,
                    entity_type=entity_type,
                    func=func,
                    position_range=position_range,
                    relative_position=bool(params.get("relative_position", True)),
                    relative_rotation=bool(params.get("relative_rotation", False)),
                    init_pos=find_entity_init_pos(gym_config, uid, entity_type),
                )
            )
    return found


def find_ik_checks(action_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect IK validators declared in the action graph configuration."""
    checks: list[dict[str, Any]] = []
    for scope, nodes in action_config.get("node", {}).items():
        for node in nodes:
            if len(node) != 1:
                continue
            node_name = next(iter(node))
            node_cfg = node[node_name]
            kwargs = node_cfg.get("kwargs", {})
            for affordance_info in kwargs.get("affordance_infos", []):
                for validator in affordance_info.get("valid_funcs_name_kwargs_proc", []):
                    if validator.get("name") != "get_ik_ret":
                        continue
                    validator_kwargs = validator.get("kwargs", {})
                    checks.append(
                        {
                            "scope": scope,
                            "node": node_name,
                            "src_key": affordance_info.get("src_key"),
                            "dst_key": affordance_info.get("dst_key"),
                            "control_part": validator_kwargs.get("control_part"),
                            "qpos_seed": validator_kwargs.get("qpos_seed"),
                        }
                    )
    return checks


def print_events(events: list[PoseRandomizationEvent]) -> None:
    if not events:
        print("No supported pose randomization events with position_range found.")
        return
    print("Supported pose randomization events:")
    for event in events:
        print(
            f"  {event.name}: type={event.entity_type}, uid={event.uid}, "
            f"func={event.func}, "
            f"position_range={event.position_range}, "
            f"relative_position={event.relative_position}, "
            f"init_pos={event.init_pos}"
        )


def select_event(
    events: list[PoseRandomizationEvent],
    uid: str | None,
    event_name: str | None,
) -> PoseRandomizationEvent:
    if event_name:
        matches = [event for event in events if event.name == event_name]
        if not matches:
            raise ValueError(f"Event '{event_name}' was not found in gym_config.")
        return matches[0]

    if uid:
        matches = [event for event in events if event.uid == uid]
        if not matches:
            raise ValueError(f"No rigid randomization event found for uid '{uid}'.")
        if len(matches) > 1:
            names = ", ".join(event.name for event in matches)
            raise ValueError(
                f"Multiple events found for uid '{uid}': {names}. Use --event."
            )
        return matches[0]

    if len(events) == 1:
        return events[0]

    print_events(events)
    raise ValueError("Please specify --uid or --event.")


def parse_axes(args_axes: list[str] | None, low: np.ndarray, high: np.ndarray) -> list[int]:
    if args_axes is not None:
        axes: list[int] = []
        for axis in args_axes:
            key = axis.lower()
            if key not in AXIS_TO_INDEX:
                raise ValueError(f"Invalid axis '{axis}'. Use x/y/z or 0/1/2.")
            axes.append(AXIS_TO_INDEX[key])
        return sorted(set(axes))

    variable_axes = [idx for idx in range(3) if abs(float(high[idx] - low[idx])) > 1e-9]
    return variable_axes or [0, 1, 2]


def expand_grid_size(grid_size: list[int], axes: list[int]) -> list[int]:
    if len(grid_size) == 1:
        sizes = grid_size * len(axes)
    elif len(grid_size) == len(axes):
        sizes = grid_size
    elif len(grid_size) == 3:
        sizes = [grid_size[axis] for axis in axes]
    else:
        raise ValueError(
            "--grid-size must contain one value, one per scanned axis, or three values."
        )
    if any(size < 1 for size in sizes):
        raise ValueError("--grid-size values must be >= 1.")
    return sizes


def generate_candidate_points(
    low: np.ndarray,
    high: np.ndarray,
    axes: list[int],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], dict[str, float]]:
    center = (low + high) / 2.0
    spacing_by_axis = {INDEX_TO_AXIS[axis]: 0.0 for axis in axes}

    if args.sample_mode == "random":
        rng = np.random.default_rng(args.seed)
        points = []
        for _ in range(args.samples):
            point = center.copy()
            sampled = rng.uniform(low[axes], high[axes])
            point[axes] = sampled
            points.append(point)
        return points, spacing_by_axis

    sizes = expand_grid_size(args.grid_size, axes)
    values_per_axis = []
    for axis, size in zip(axes, sizes):
        if size == 1:
            values = np.array([(low[axis] + high[axis]) / 2.0], dtype=float)
            spacing = 0.0
        else:
            values = np.linspace(low[axis], high[axis], num=size, dtype=float)
            spacing = float(values[1] - values[0])
        values_per_axis.append(values)
        spacing_by_axis[INDEX_TO_AXIS[axis]] = spacing

    points = []
    for combo in product(*values_per_axis):
        point = center.copy()
        for axis, value in zip(axes, combo):
            point[axis] = value
        points.append(point)
    return points, spacing_by_axis


def range_to_actual(
    low: np.ndarray,
    high: np.ndarray,
    event: PoseRandomizationEvent,
) -> tuple[np.ndarray, np.ndarray]:
    if not event.relative_position:
        return low.copy(), high.copy()
    if event.init_pos is None:
        raise ValueError(
            f"Event '{event.name}' uses relative_position=true, but init_pos for "
            f"uid '{event.uid}' was not found in gym_config."
        )
    init_pos = np.asarray(event.init_pos, dtype=float)
    return low + init_pos, high + init_pos


def point_to_actual(point: np.ndarray, event: PoseRandomizationEvent) -> np.ndarray:
    if not event.relative_position:
        return point.copy()
    if event.init_pos is None:
        raise ValueError(
            f"Event '{event.name}' uses relative_position=true, but init_pos for "
            f"uid '{event.uid}' was not found in gym_config."
        )
    return point + np.asarray(event.init_pos, dtype=float)


def actual_range_to_config(
    low: np.ndarray,
    high: np.ndarray,
    event: PoseRandomizationEvent,
) -> tuple[np.ndarray, np.ndarray]:
    if not event.relative_position:
        return low.copy(), high.copy()
    if event.init_pos is None:
        raise ValueError(
            f"Event '{event.name}' uses relative_position=true, but init_pos for "
            f"uid '{event.uid}' was not found in gym_config."
        )
    init_pos = np.asarray(event.init_pos, dtype=float)
    return low - init_pos, high - init_pos


def filter_gym_config_for_analysis(
    gym_config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    """Remove events that are irrelevant to IK reachability analysis."""
    filtered_config = deepcopy(gym_config)
    removed_events: list[str] = []

    if not args.filter_distractor_events:
        return filtered_config, removed_events

    events = filtered_config.get("env", {}).get("events", {})
    for event_name, event_cfg in list(events.items()):
        func_name = str(event_cfg.get("func", "")).lower()
        params_text = json.dumps(event_cfg.get("params", {}), sort_keys=True).lower()
        if (
            "distractor" in event_name.lower()
            or "distractor" in func_name
            or "distractor" in params_text
        ):
            events.pop(event_name)
            removed_events.append(event_name)

    return filtered_config, removed_events


def import_runtime() -> None:
    import embodied_challenge  # noqa: F401
    import embodichain.lab.gym.utils.gym_utils as gym_utils

    for module_name in (
        "embodied_challenge.managers.actions",
        "embodied_challenge.managers.datasets",
        "embodied_challenge.managers.events",
        "embodied_challenge.managers.observations",
    ):
        if module_name not in gym_utils.DEFAULT_MANAGER_MODULES:
            gym_utils.DEFAULT_MANAGER_MODULES.append(module_name)


def make_env(args: argparse.Namespace):
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import (
        config_to_cfg,
        merge_args_with_gym_config,
    )
    from embodichain.lab.sim import SimulationManagerCfg

    raw_gym_config = load_json(args.gym_config)
    filtered_gym_config, removed_events = filter_gym_config_for_analysis(
        raw_gym_config, args
    )
    if removed_events:
        print(
            "Filtered distractor events before env creation: "
            + ", ".join(removed_events)
        )

    gym_config = merge_args_with_gym_config(args, filtered_gym_config)
    env_cfg = config_to_cfg(gym_config)
    env_cfg.filter_visual_rand = args.filter_visual_rand
    env_cfg.filter_dataset_saving = args.filter_dataset_saving
    if args.preview:
        env_cfg.filter_dataset_saving = True

    action_config = {}
    if args.action_config is not None:
        action_config = load_json(args.action_config)
        action_config["action_config"] = action_config

    env_cfg.sim_cfg = SimulationManagerCfg(
        headless=gym_config["headless"],
        sim_device=gym_config["device"],
        enable_rt=gym_config["enable_rt"],
        gpu_id=gym_config["gpu_id"],
        arena_space=gym_config["arena_space"],
    )

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    return env


def unwrap_env(env):
    return getattr(env, "unwrapped", env)


def set_event_position_range(
    env,
    event_name: str,
    low: Iterable[float],
    high: Iterable[float],
    relative_position: bool | None = None,
) -> None:
    base_env = unwrap_env(env)
    if base_env.event_manager is None:
        raise RuntimeError("Environment has no event_manager.")
    event_cfg = base_env.event_manager.get_functor_cfg(event_name)
    if event_cfg is None:
        raise RuntimeError(f"Event '{event_name}' was not found in event_manager.")
    event_cfg.params["position_range"] = [
        [float(v) for v in low],
        [float(v) for v in high],
    ]
    if relative_position is not None:
        event_cfg.params["relative_position"] = relative_position
    base_env.event_manager.set_functor_cfg(event_name, event_cfg)


def tensor_bool(value: Any) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(value.detach().cpu().reshape(-1).all().item())
    except ModuleNotFoundError:
        pass
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1).all())
    if isinstance(value, (list, tuple)):
        return bool(np.asarray(value).reshape(-1).all())
    return bool(value)


def iter_action_steps(action_list: Any):
    try:
        import torch

        if isinstance(action_list, torch.Tensor):
            for idx in range(action_list.shape[0]):
                yield action_list[idx]
            return
    except ModuleNotFoundError:
        pass
    for action in action_list:
        yield action


def rollout_actions(env, action_list: Any, max_steps: int | None) -> bool:
    base_env = unwrap_env(env)
    last_info: dict[str, Any] = {}

    for step_idx, action in enumerate(iter_action_steps(action_list)):
        if max_steps is not None and step_idx >= max_steps:
            break
        step_ret = env.step(action)
        if len(step_ret) == 5:
            _, _, terminated, truncated, info = step_ret
            done = tensor_bool(terminated) or tensor_bool(truncated)
        else:
            _, _, done, info = step_ret
            done = tensor_bool(done)
        last_info = info
        if done:
            break

    if hasattr(base_env, "is_task_success"):
        return tensor_bool(base_env.is_task_success())
    if "success" in last_info:
        return tensor_bool(last_info["success"])
    return False


def run_trial(
    env,
    seed: int,
    args: argparse.Namespace,
) -> TrialResult:
    base_env = unwrap_env(env)
    try:
        if hasattr(base_env, "affordance_datas"):
            base_env.affordance_datas = {}
        env.reset(seed=seed)
        action_list = base_env.create_demo_action_list()
        if action_list is None:
            return TrialResult(
                generated=False,
                rollout_success=False if args.rollout else None,
                error="create_demo_action_list returned None",
            )
        if len(action_list) == 0:
            return TrialResult(
                generated=False,
                rollout_success=False if args.rollout else None,
                error="create_demo_action_list returned an empty action list",
            )
        if args.rollout:
            rollout_success = rollout_actions(env, action_list, args.max_rollout_steps)
            return TrialResult(generated=True, rollout_success=rollout_success)
        return TrialResult(generated=True, rollout_success=None)
    except Exception as exc:  # noqa: BLE001 - failures are data for this analyzer.
        if args.verbose_failures:
            traceback.print_exc()
        return TrialResult(generated=False, rollout_success=False if args.rollout else None, error=str(exc))


def evaluate_fixed_point(
    env,
    event: PoseRandomizationEvent,
    point: np.ndarray,
    point_index: int,
    args: argparse.Namespace,
) -> PointResult:
    actual_point = point_to_actual(point, event)
    set_event_position_range(
        env,
        event.name,
        actual_point,
        actual_point,
        relative_position=False,
    )
    generation_successes = 0
    rollout_successes = 0
    errors: list[str] = []

    for trial_idx in range(args.trials_per_point):
        seed = args.seed + point_index * max(args.trials_per_point, 1) + trial_idx
        result = run_trial(env, seed=seed, args=args)
        generation_successes += int(result.generated)
        if args.rollout:
            rollout_successes += int(bool(result.generated and result.rollout_success))
        if result.error:
            errors.append(result.error)

    attempts = args.trials_per_point
    metric_successes = rollout_successes if args.rollout else generation_successes
    return PointResult(
        index=point_index,
        point=[float(v) for v in point],
        actual_point=[float(v) for v in actual_point],
        attempts=attempts,
        generation_successes=generation_successes,
        rollout_successes=rollout_successes if args.rollout else None,
        success_rate=metric_successes / attempts if attempts else 0.0,
        generation_success_rate=generation_successes / attempts if attempts else 0.0,
        errors=errors[:5],
    )


def evaluate_range(
    env,
    event: PoseRandomizationEvent,
    low: np.ndarray,
    high: np.ndarray,
    attempts: int,
    seed_offset: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if attempts <= 0:
        return {"attempts": 0, "successes": 0, "success_rate": None}

    actual_low, actual_high = range_to_actual(low, high, event)
    set_event_position_range(
        env,
        event.name,
        actual_low,
        actual_high,
        relative_position=False,
    )
    successes = 0
    generation_successes = 0
    errors: list[str] = []
    for idx in range(attempts):
        result = run_trial(env, seed=args.seed + seed_offset + idx, args=args)
        generation_successes += int(result.generated)
        if args.rollout:
            successes += int(bool(result.generated and result.rollout_success))
        else:
            successes += int(result.generated)
        if result.error:
            errors.append(result.error)

    return {
        "attempts": attempts,
        "successes": successes,
        "success_rate": successes / attempts,
        "generation_successes": generation_successes,
        "generation_success_rate": generation_successes / attempts,
        "errors": errors[:10],
    }


def make_recommended_range(
    point_results: list[PointResult],
    original_low: np.ndarray,
    original_high: np.ndarray,
    axes: list[int],
    spacing_by_axis: dict[str, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    good_points = np.asarray(
        [
            result.point
            for result in point_results
            if result.success_rate >= args.point_success_threshold
        ],
        dtype=float,
    )
    if good_points.size == 0:
        raise RuntimeError(
            "No safe points found. Try lowering --point-success-threshold, "
            "increasing --samples/--grid-size, or checking the action_config."
        )

    rec_low = original_low.copy()
    rec_high = original_high.copy()
    trim = float(args.trim_quantile)
    if trim < 0.0 or trim >= 0.5:
        raise ValueError("--trim-quantile must be in [0, 0.5).")

    for axis in axes:
        axis_values = good_points[:, axis]
        lo = float(np.quantile(axis_values, trim))
        hi = float(np.quantile(axis_values, 1.0 - trim))

        spacing = spacing_by_axis.get(INDEX_TO_AXIS[axis], 0.0)
        if args.sample_mode == "grid" and spacing > 0.0:
            lo -= spacing / 2.0
            hi += spacing / 2.0

        lo -= args.margin
        hi += args.margin
        rec_low[axis] = max(float(original_low[axis]), lo)
        rec_high[axis] = min(float(original_high[axis]), hi)

    return rec_low, rec_high, len(good_points)


def shrink_range(
    low: np.ndarray,
    high: np.ndarray,
    axes: list[int],
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < factor < 1.0:
        raise ValueError("--shrink-factor must be between 0 and 1.")
    new_low = low.copy()
    new_high = high.copy()
    center = (low + high) / 2.0
    half_width = (high - low) * factor / 2.0
    for axis in axes:
        new_low[axis] = center[axis] - half_width[axis]
        new_high[axis] = center[axis] + half_width[axis]
    return new_low, new_high


def write_csv(path: str | Path, point_results: list[PointResult]) -> None:
    fieldnames = [
        "index",
        "x",
        "y",
        "z",
        "actual_x",
        "actual_y",
        "actual_z",
        "attempts",
        "generation_successes",
        "rollout_successes",
        "success_rate",
        "generation_success_rate",
        "errors",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in point_results:
            row = {
                "index": result.index,
                "x": result.point[0],
                "y": result.point[1],
                "z": result.point[2],
                "actual_x": result.actual_point[0],
                "actual_y": result.actual_point[1],
                "actual_z": result.actual_point[2],
                "attempts": result.attempts,
                "generation_successes": result.generation_successes,
                "rollout_successes": result.rollout_successes,
                "success_rate": result.success_rate,
                "generation_success_rate": result.generation_success_rate,
                "errors": " | ".join(result.errors),
            }
            writer.writerow(row)


def resolve_plot_path(args: argparse.Namespace) -> str | None:
    if not args.save_plot:
        return None
    if args.plot:
        return args.plot
    if args.output:
        return str(Path(args.output).with_suffix(".png"))
    return "spawn_range_analysis.png"


def draw_2d_bbox(ax, low: np.ndarray, high: np.ndarray, axes: list[int], **kwargs) -> None:
    from matplotlib.patches import Rectangle

    vertical_axis, horizontal_axis = axes
    width = float(high[horizontal_axis] - low[horizontal_axis])
    height = float(high[vertical_axis] - low[vertical_axis])
    rect = Rectangle(
        (float(low[horizontal_axis]), float(low[vertical_axis])),
        width,
        height,
        fill=False,
        **kwargs,
    )
    ax.add_patch(rect)


def draw_3d_bbox(ax, low: np.ndarray, high: np.ndarray, axes: list[int], **kwargs) -> None:
    corners = []
    for x in (low[axes[0]], high[axes[0]]):
        for y in (low[axes[1]], high[axes[1]]):
            for z in (low[axes[2]], high[axes[2]]):
                corners.append((float(x), float(y), float(z)))

    def differs_by_one_bit(i: int, j: int) -> bool:
        return bin(i ^ j).count("1") == 1

    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if differs_by_one_bit(i, j):
                xs = [corners[i][0], corners[j][0]]
                ys = [corners[i][1], corners[j][1]]
                zs = [corners[i][2], corners[j][2]]
                ax.plot(xs, ys, zs, **kwargs)


def save_spawn_range_plot(
    path: str | Path,
    point_results: list[PointResult],
    original_range: list[list[float]],
    recommended_range: list[list[float]],
    axes: list[int],
    target_event: PoseRandomizationEvent,
    args: argparse.Namespace,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    plot_path = Path(path)
    if plot_path.parent != Path("."):
        plot_path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray([result.actual_point for result in point_results], dtype=float)
    success_mask = np.asarray(
        [
            result.success_rate >= args.point_success_threshold
            for result in point_results
        ],
        dtype=bool,
    )
    original_low = np.asarray(original_range[0], dtype=float)
    original_high = np.asarray(original_range[1], dtype=float)
    recommended_low = np.asarray(recommended_range[0], dtype=float)
    recommended_high = np.asarray(recommended_range[1], dtype=float)

    success_label = f"success >= {args.point_success_threshold:g}"
    fail_label = f"success < {args.point_success_threshold:g}"

    if len(axes) >= 3:
        plot_axes = axes[:3]
        fig = plt.figure(figsize=(8.5, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        if (~success_mask).any():
            ax.scatter(
                points[~success_mask, plot_axes[0]],
                points[~success_mask, plot_axes[1]],
                points[~success_mask, plot_axes[2]],
                c="#d62728",
                marker="x",
                s=42,
                label=fail_label,
                depthshade=False,
            )
        if success_mask.any():
            ax.scatter(
                points[success_mask, plot_axes[0]],
                points[success_mask, plot_axes[1]],
                points[success_mask, plot_axes[2]],
                c="#2ca02c",
                marker="o",
                s=36,
                edgecolors="black",
                linewidths=0.4,
                label=success_label,
                depthshade=False,
            )
        draw_3d_bbox(
            ax,
            original_low,
            original_high,
            plot_axes,
            color="#8c8c8c",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        draw_3d_bbox(
            ax,
            recommended_low,
            recommended_high,
            plot_axes,
            color="#1f77b4",
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
        )
        ax.set_xlabel(f"{INDEX_TO_AXIS[plot_axes[0]]} position (m)")
        ax.set_ylabel(f"{INDEX_TO_AXIS[plot_axes[1]]} position (m)")
        ax.set_zlabel(f"{INDEX_TO_AXIS[plot_axes[2]]} position (m)")
        original_handle = mlines.Line2D(
            [], [], color="#8c8c8c", linestyle="--", label="original bbox"
        )
        recommended_handle = mlines.Line2D(
            [], [], color="#1f77b4", linewidth=2.0, label="recommended bbox"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [original_handle, recommended_handle], labels + ["original bbox", "recommended bbox"])
    elif len(axes) == 2:
        plot_axes = axes
        fig, ax = plt.subplots(figsize=(8.0, 6.5))
        vertical_axis, horizontal_axis = plot_axes
        if (~success_mask).any():
            ax.scatter(
                points[~success_mask, horizontal_axis],
                points[~success_mask, vertical_axis],
                c="#d62728",
                marker="x",
                s=52,
                label=fail_label,
            )
        if success_mask.any():
            ax.scatter(
                points[success_mask, horizontal_axis],
                points[success_mask, vertical_axis],
                c="#2ca02c",
                marker="o",
                s=42,
                edgecolors="black",
                linewidths=0.4,
                label=success_label,
            )
        draw_2d_bbox(
            ax,
            original_low,
            original_high,
            plot_axes,
            edgecolor="#8c8c8c",
            linestyle="--",
            linewidth=1.4,
            label="original bbox",
        )
        draw_2d_bbox(
            ax,
            recommended_low,
            recommended_high,
            plot_axes,
            edgecolor="#1f77b4",
            linestyle="-",
            linewidth=2.2,
            label="recommended bbox",
        )
        ax.set_xlabel(f"{INDEX_TO_AXIS[horizontal_axis]} position (m)")
        ax.set_ylabel(f"{INDEX_TO_AXIS[vertical_axis]} position (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        ax.grid(True, alpha=0.25)
    else:
        axis = axes[0]
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        x_values = points[:, axis]
        y_values = np.asarray([result.success_rate for result in point_results])
        if (~success_mask).any():
            ax.scatter(
                x_values[~success_mask],
                y_values[~success_mask],
                c="#d62728",
                marker="x",
                s=52,
                label=fail_label,
            )
        if success_mask.any():
            ax.scatter(
                x_values[success_mask],
                y_values[success_mask],
                c="#2ca02c",
                marker="o",
                s=42,
                edgecolors="black",
                linewidths=0.4,
                label=success_label,
            )
        ax.axvspan(
            original_low[axis],
            original_high[axis],
            color="#8c8c8c",
            alpha=0.12,
            label="original range",
        )
        ax.axvspan(
            recommended_low[axis],
            recommended_high[axis],
            color="#1f77b4",
            alpha=0.18,
            label="recommended range",
        )
        ax.axhline(
            args.point_success_threshold,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.2,
            label="point threshold",
        )
        ax.set_xlabel(f"{INDEX_TO_AXIS[axis]} position (m)")
        ax.set_ylabel("success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{target_event.uid} spawn feasibility: {target_event.name}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def patch_gym_config(
    gym_config_path: str | Path,
    event_name: str,
    recommended_range: list[list[float]],
    output_path: str | Path,
) -> None:
    gym_config = load_json(gym_config_path)
    gym_config["env"]["events"][event_name]["params"]["position_range"] = recommended_range
    dump_json(output_path, gym_config)


def build_paste_snippet(
    gym_config: dict[str, Any],
    event_name: str,
    recommended_range: list[list[float]],
) -> dict[str, Any]:
    event_cfg = json.loads(json.dumps(gym_config["env"]["events"][event_name]))
    event_cfg["params"]["position_range"] = recommended_range
    return {event_name: event_cfg}


def main() -> None:
    args = parse_args()
    gym_config = load_json(args.gym_config)
    events = find_pose_randomization_events(gym_config)

    if args.list_events:
        print_events(events)
        return

    if args.action_config is None:
        raise ValueError("--action_config is required unless --list-events is used.")
    if args.num_envs != 1:
        raise ValueError("Expert-demo generation currently supports --num_envs 1.")
    require_numpy()
    action_config = load_json(args.action_config)
    ik_checks = find_ik_checks(action_config)

    target_event = select_event(events, uid=args.uid, event_name=args.event)
    original_range = np.asarray(target_event.position_range, dtype=float)
    if original_range.shape != (2, 3):
        raise ValueError(
            f"Expected position_range shape [2, 3], got {target_event.position_range}."
        )
    original_low = original_range[0]
    original_high = original_range[1]
    original_actual_low, original_actual_high = range_to_actual(
        original_low,
        original_high,
        target_event,
    )
    axes = parse_axes(args.axes, original_low, original_high)
    axis_names = [INDEX_TO_AXIS[axis] for axis in axes]

    points, spacing_by_axis = generate_candidate_points(
        original_low,
        original_high,
        axes,
        args,
    )

    print(
        f"Analyzing {target_event.entity_type} event '{target_event.name}' "
        f"for uid '{target_event.uid}' "
        f"over axes {axis_names} with {len(points)} candidate points."
    )
    print(f"Original config position_range: {target_event.position_range}")
    if target_event.relative_position:
        print(f"Entity init_pos: {target_event.init_pos}")
        print(
            "Original actual position_range: "
            f"{[[float(v) for v in original_actual_low], [float(v) for v in original_actual_high]]}"
        )
    print(f"IK checks found in action_config: {len(ik_checks)}")

    import_runtime()
    env = make_env(args)
    point_results: list[PointResult] = []
    validation_history: list[dict[str, Any]] = []

    try:
        for idx, point in enumerate(points):
            result = evaluate_fixed_point(env, target_event, point, idx, args)
            point_results.append(result)
            point_text = ", ".join(f"{v:.5f}" for v in result.actual_point)
            print(
                f"[{idx + 1}/{len(points)}] point=[{point_text}] "
                f"success_rate={result.success_rate:.3f} "
                f"generation={result.generation_success_rate:.3f}"
            )
            if result.success_rate == 0.0 and result.errors:
                print(f"    error: {result.errors[0]}")

        rec_low, rec_high, good_point_count = make_recommended_range(
            point_results,
            original_low,
            original_high,
            axes,
            spacing_by_axis,
            args,
        )

        if args.validation_samples > 0:
            for iteration in range(args.max_shrink_iterations + 1):
                validation = evaluate_range(
                    env,
                    target_event,
                    rec_low,
                    rec_high,
                    attempts=args.validation_samples,
                    seed_offset=100_000 + iteration * args.validation_samples,
                    args=args,
                )
                validation["iteration"] = iteration
                validation["position_range"] = [
                    [float(v) for v in rec_low],
                    [float(v) for v in rec_high],
                ]
                validation_actual_low, validation_actual_high = range_to_actual(
                    rec_low,
                    rec_high,
                    target_event,
                )
                validation["actual_position_range"] = [
                    [float(v) for v in validation_actual_low],
                    [float(v) for v in validation_actual_high],
                ]
                validation_history.append(validation)
                rate = validation["success_rate"]
                print(
                    f"Validation {iteration}: success_rate={rate:.3f}, "
                    f"range={validation['position_range']}"
                )
                if rate >= args.target_success_rate:
                    break
                if iteration < args.max_shrink_iterations:
                    rec_low, rec_high = shrink_range(
                        rec_low,
                        rec_high,
                        axes,
                        args.shrink_factor,
                    )

        recommended_range = [
            [float(v) for v in rec_low],
            [float(v) for v in rec_high],
        ]
        recommended_actual_low, recommended_actual_high = range_to_actual(
            rec_low,
            rec_high,
            target_event,
        )
        recommended_actual_range = [
            [float(v) for v in recommended_actual_low],
            [float(v) for v in recommended_actual_high],
        ]
        final_validation = validation_history[-1] if validation_history else None
        plot_path = resolve_plot_path(args)
        saved_plot_path = None
        plot_error = None
        if plot_path is not None:
            try:
                saved_plot_path = save_spawn_range_plot(
                    path=plot_path,
                    point_results=point_results,
                    original_range=[
                        [float(v) for v in original_actual_low],
                        [float(v) for v in original_actual_high],
                    ],
                    recommended_range=recommended_actual_range,
                    axes=axes,
                    target_event=target_event,
                    args=args,
                )
            except ModuleNotFoundError as exc:
                plot_error = str(exc)
                print(f"Plot skipped because matplotlib is unavailable: {exc}")
        paste_snippet = build_paste_snippet(
            gym_config,
            target_event.name,
            recommended_range,
        )
        report = {
            "gym_config": str(args.gym_config),
            "action_config": str(args.action_config),
            "uid": target_event.uid,
            "event": target_event.name,
            "entity_type": target_event.entity_type,
            "event_func": target_event.func,
            "relative_position": target_event.relative_position,
            "init_pos": target_event.init_pos,
            "metric": (
                "expert_generation_and_rollout_success"
                if args.rollout
                else "expert_generation_success"
            ),
            "axes": axis_names,
            "original_position_range": target_event.position_range,
            "original_actual_position_range": [
                [float(v) for v in original_actual_low],
                [float(v) for v in original_actual_high],
            ],
            "recommended_position_range": recommended_range,
            "recommended_actual_position_range": recommended_actual_range,
            "target_success_rate": args.target_success_rate,
            "point_success_threshold": args.point_success_threshold,
            "ik_checks": ik_checks,
            "candidate_point_count": len(points),
            "good_point_count": good_point_count,
            "validation": final_validation,
            "validation_history": validation_history,
            "plot": saved_plot_path,
            "plot_error": plot_error,
            "paste_snippet": paste_snippet,
            "points": [asdict(result) for result in point_results],
        }

        print("\nRecommended position_range:")
        print(json.dumps(recommended_range, indent=2))
        if target_event.relative_position:
            print("\nRecommended actual position_range:")
            print(json.dumps(recommended_actual_range, indent=2))
        if final_validation is not None:
            print(
                "Final validation success_rate: "
                f"{final_validation['success_rate']:.3f} "
                f"({final_validation['successes']}/{final_validation['attempts']})"
            )
        print("\nReady-to-paste event snippet:")
        print(json.dumps(paste_snippet, indent=2))

        if saved_plot_path:
            print(f"Wrote plot: {saved_plot_path}")
        if args.output:
            dump_json(args.output, report)
            print(f"Wrote JSON report: {args.output}")
        if args.csv:
            write_csv(args.csv, point_results)
            print(f"Wrote CSV point results: {args.csv}")
        if args.write_gym_config:
            patch_gym_config(
                args.gym_config,
                target_event.name,
                recommended_range,
                args.write_gym_config,
            )
            print(f"Wrote patched gym_config: {args.write_gym_config}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
