"""Replay native expert controller actions from a seeded or recorded reset."""

import argparse
import math
from pathlib import Path

import gymnasium as gym
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.demo import DemoSegment, execute_demo_episode
from embodichain.lab.gym.envs.expert_trajectory import (
    EXPERT_TRAJECTORY_SCHEMA_VERSION,
    build_expert_action_spec,
)
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils import gym_utils
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.gym.utils.trajectory_state import (
    capture_trajectory_state,
    restore_trajectory_state,
)
from embodichain.lab.sim import SimulationManager

from scripts.eval_policy import prepare_episode_reset


def verify_initial_state(env, expected):
    """Check scene geometry before issuing any recorded action."""
    expected = expected.to(env.device)
    observed = expected.clone()
    env_ids = torch.arange(env.num_envs, device=env.device)
    capture_trajectory_state(env, observed, env_ids)
    for key, target in expected.items(include_nested=True, leaves_only=True):
        if key[-1] == "qpos":
            matches = torch.allclose(observed[key], target, atol=0.002, rtol=0)
        elif key[-1] in ("pose", "root_pose"):
            pose = observed[key]
            distance = torch.linalg.vector_norm(pose[..., :3] - target[..., :3], dim=-1)
            alignment = (pose[..., 3:] * target[..., 3:]).sum(dim=-1).abs()
            matches = bool(((distance < 0.003) & (alignment > 0.999)).all())
        else:
            continue
        if not matches:
            raise ValueError(
                f"The seeded reset disagrees with the recorded initial {key!r}."
            )


def controller_actions(commands, spec):
    for frame in commands.unbind(dim=1):
        targets = {"qpos": frame[:, slice(*spec.qpos_slice)]}
        if spec.qvel_slice is not None:
            targets["qvel"] = frame[:, slice(*spec.qvel_slice)]
        yield ControllerAction(TensorDict(targets, batch_size=[1]))


def recorded_segments(commands, spec, episode):
    """Use the source episode's instructions and boundaries during replay."""
    cursor = 0
    segments = []
    if not episode["success"]:
        raise ValueError("Recording a replay requires a successful source episode.")
    for segment in episode["segments"]:
        start, end = segment["start_step"], segment["end_step"]
        if start != cursor or not start < end <= commands.shape[1] or not segment["success"]:
            raise ValueError("The recorded segment spans do not describe a successful episode.")
        segments.append(
            DemoSegment(
                name=segment["name"],
                target_uid=segment.get("target_uid"),
                instruction=segment["instruction"],
                instruction_source=segment.get("instruction_source"),
                actions=controller_actions(commands[:, start:end], spec),
                progress_total_steps=end - start,
            )
        )
        cursor = end
    if cursor != commands.shape[1]:
        raise ValueError("The recorded segments do not cover every controller action.")
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    gym_utils.add_env_launcher_args_to_parser(parser, require_gym_config=True)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--dataset-dir", dest="expansion_dataset_dir")
    parser.add_argument(
        "--recorded-reset", action="store_true",
        help="Restore the recorded initial robot and object state before dynamic replay.",
    )
    args = parser.parse_args()
    args.num_envs = 1
    args.filter_dataset_saving = args.expansion_dataset_dir is None
    data = gym_utils.load_trajectory(args.trajectory)
    meta = data["meta"]
    if (
        meta["num_envs"] != 1
        or meta.get("action_kind") != "expert"
        or meta.get("expert_trajectory_schema_version")
        != EXPERT_TRAJECTORY_SCHEMA_VERSION
    ):
        raise ValueError("Seeded replay requires one episode of native expert actions.")
    episode = meta["demo_episodes"][0]
    seed = episode.get("seed")
    if type(seed) is not int:
        raise ValueError("The native episode has no recorded reset seed.")
    spec = build_expert_action_spec(
        joint_names=meta["joint_names"], joint_command_mode=meta["joint_command_mode"]
    )

    discover_task_packages()
    execute_init_hooks()
    cfg, selection, _ = gym_utils.build_env_cfg_from_args(args)
    env = gym.make(selection["id"], cfg=cfg)
    task = env.unwrapped
    try:
        runtime_joints = tuple(
            task.robot.joint_names[index] for index in task.active_joint_ids
        )
        if spec.joint_names != runtime_joints or not math.isclose(
            meta["step_dt"], task.step_dt, rel_tol=1e-6
        ):
            raise ValueError("The recorded controller does not match the scene.")
        prepare_episode_reset(env, seed)
        env.reset(seed=seed, options={"save_data": False})
        initial_state = data["states"][:, 0]
        if args.recorded_reset:
            restore_trajectory_state(task, initial_state.to(task.device))
        verify_initial_state(task, initial_state)

        commands = data["actions"].to(task.device)
        if commands.shape[-1] != spec.width or not torch.isfinite(commands).all():
            raise ValueError("The recorded actions do not match the controller layout.")
        length = int(meta["lengths"][0])
        commands = commands[:, :length]
        if args.expansion_dataset_dir is not None:
            result = execute_demo_episode(
                env, segments=recorded_segments(commands, spec, episode),
                instruction=episode["instruction"],
            )
            completed = result.length
            accepted = completed == length and all(result.success)
            info = task.get_info()
        else:
            completed = 0
            info = {}
            for command in controller_actions(commands, spec):
                _, _, terminated, truncated, info = env.step(command)
                completed += 1
                if bool((terminated | truncated).any()):
                    break
            accepted = completed == length
        accepted = accepted and bool(task.is_task_success().all())
        print(f"Seed {seed} replayed {completed}/{length} actions. Success={accepted}")
        if info.get("metrics"):
            print(info["metrics"])
        if accepted and args.expansion_dataset_dir is not None:
            env.reset(options={"save_data": True, "commit_env_ids": [0]})
        return 0 if accepted else 1
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    raise SystemExit(main())
