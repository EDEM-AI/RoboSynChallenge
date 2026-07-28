# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Trajectory replay compatibility helpers for RoboSynChallenge."""

from __future__ import annotations

from os import PathLike
from typing import Any, Literal

import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.wrapper import ReplayWrapper
from embodichain.lab.scripts.run_env import replay as replay_native_trajectory
from embodichain.lab.scripts.run_env import replay_auto, replay_control
from embodichain.utils.logger import log_info, log_warning

__all__ = ["convert_state_action_trajectory", "replay_trajectory"]

ReplayMode = Literal["kinematic", "dynamic", "control"]


def _as_batched_trajectory(value: Any, key: str) -> torch.Tensor:
    """Return a legacy trajectory tensor in ``(num_envs, steps, dim)`` form."""
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            f"Legacy trajectory {key!r} must have shape (steps, dim) or "
            f"(num_envs, steps, dim), got {tuple(tensor.shape)}."
        )
    if tensor.shape[0] != 1:
        raise ValueError(
            "Legacy state/action replay currently expects one recorded environment, "
            f"got {tensor.shape[0]}."
        )
    return tensor


def _valid_state_prefix(states: torch.Tensor) -> int:
    """Return the number of consecutive finite state frames."""
    finite_steps = torch.isfinite(states).all(dim=-1).all(dim=0)
    invalid = (~finite_steps).nonzero(as_tuple=False)
    if len(invalid) == 0:
        return int(states.shape[1])
    return int(invalid[0].item())


def _denormalize_eef_states(states: torch.Tensor, env: Any) -> torch.Tensor:
    """Convert normalized CobotMagic gripper observations back to joint qpos."""
    result = states.clone()
    active_joint_ids = list(env.active_joint_ids)
    global_to_active = {
        joint_id: active_idx for active_idx, joint_id in enumerate(active_joint_ids)
    }

    eef_active_indices: list[int] = []
    for control_part in ("left_eef", "right_eef"):
        for joint_id in env.robot.get_joint_ids(name=control_part, remove_mimic=True):
            if joint_id in global_to_active:
                eef_active_indices.append(global_to_active[joint_id])

    if not eef_active_indices:
        return result

    active_indices = torch.as_tensor(eef_active_indices, dtype=torch.long)
    robot_joint_ids = torch.as_tensor(
        [active_joint_ids[idx] for idx in eef_active_indices],
        device=env.device,
        dtype=torch.long,
    )
    limits = (
        env.robot.body_data.qpos_limits[0, robot_joint_ids]
        .detach()
        .to(device="cpu", dtype=result.dtype)
    )
    low = limits[:, 0]
    high = limits[:, 1]
    result[..., active_indices] = low + result[..., active_indices] * (high - low)
    return result


def _expand_active_qpos(states: torch.Tensor, env: Any) -> torch.Tensor:
    """Expand active-joint qpos into the robot's complete qpos representation."""
    num_envs, num_steps, action_dim = states.shape
    active_joint_ids = list(env.active_joint_ids)
    if action_dim != len(active_joint_ids):
        raise ValueError(
            f"Legacy trajectory has {action_dim} joints, but the replay environment "
            f"has {len(active_joint_ids)} active joints ({active_joint_ids})."
        )

    active_ids_tensor = torch.as_tensor(
        active_joint_ids, device=env.device, dtype=torch.long
    )
    active_limits = (
        env.robot.body_data.qpos_limits[0, active_ids_tensor]
        .detach()
        .to(device="cpu", dtype=states.dtype)
    )
    low = active_limits[:, 0]
    high = active_limits[:, 1]
    out_of_range = (states < low) | (states > high)
    if bool(out_of_range.any()):
        value_count = int(out_of_range.sum().item())
        step_count = int(out_of_range.any(dim=-1).sum().item())
        log_warning(
            f"Clamping {value_count} out-of-range legacy state values across "
            f"{step_count} frames to the robot joint limits."
        )
        states = states.clamp(low, high)

    initial_qpos = (
        env.robot.get_qpos()
        .detach()
        .to(device="cpu", dtype=states.dtype)
        .expand(num_envs, -1)
    )
    qpos = initial_qpos[:, None, :].expand(num_envs, num_steps, -1).clone()
    qpos[..., active_joint_ids] = states

    mimic_ids = list(env.robot.mimic_ids)
    mimic_parents = list(env.robot.mimic_parents)
    mimic_multipliers = list(env.robot.mimic_multipliers)
    mimic_offsets = list(env.robot.mimic_offsets)
    for mimic_id, parent_id, multiplier, offset in zip(
        mimic_ids, mimic_parents, mimic_multipliers, mimic_offsets
    ):
        if mimic_id is None or parent_id is None:
            continue
        qpos[..., mimic_id] = qpos[..., parent_id] * multiplier + offset

    return qpos


def convert_state_action_trajectory(
    data: dict[str, Any],
    env: Any,
    mode: ReplayMode = "kinematic",
) -> dict[str, Any]:
    """Convert a legacy ``state/action/reward`` file for ``ReplayWrapper``.

    The legacy format stores only the robot's 14 active CobotMagic joints. It
    does not contain drawer or rigid-object poses, so kinematic/control replay
    reproduces robot motion only. Dynamic replay feeds the recorded actions
    through physics and can therefore move scene objects.
    """
    if "state" not in data or "action" not in data:
        raise ValueError(
            "Unsupported trajectory format. Expected EmbodiChain "
            "states/actions/meta or legacy state/action/reward keys."
        )

    states = _as_batched_trajectory(data["state"], "state")
    actions = _as_batched_trajectory(data["action"], "action")
    if states.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "Legacy state/action lengths do not match: "
            f"state={tuple(states.shape)}, action={tuple(actions.shape)}."
        )
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("Legacy trajectory contains non-finite actions.")

    finite_prefix = _valid_state_prefix(states)
    if finite_prefix == 0:
        raise ValueError("Legacy trajectory has no finite state frames.")

    if mode in ("kinematic", "control"):
        num_steps = finite_prefix
        if num_steps < states.shape[1]:
            log_warning(
                f"Legacy state trajectory becomes non-finite at step {num_steps}; "
                f"replaying its {num_steps} valid frames."
            )
        states = states[:, :num_steps]
        actions = actions[:, :num_steps]
    else:
        num_steps = int(actions.shape[1])
        if finite_prefix < num_steps:
            log_warning(
                f"Legacy state trajectory becomes non-finite at step {finite_prefix}; "
                "dynamic replay will continue with the finite action trajectory."
            )
            states = states.clone()
            states[:, finite_prefix:] = states[:, finite_prefix - 1 : finite_prefix]

    physical_states = _denormalize_eef_states(states, env)
    robot_qpos = _expand_active_qpos(physical_states, env)
    root_pose = (
        env.robot.get_local_pose()
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .unsqueeze(1)
        .expand(-1, num_steps, -1)
        .clone()
    )
    robot_states = TensorDict(
        {"root_pose": root_pose, "qpos": robot_qpos},
        batch_size=[states.shape[0], num_steps],
        device="cpu",
    )
    replay_states = TensorDict(
        {"robot": robot_states},
        batch_size=[states.shape[0], num_steps],
        device="cpu",
    )
    meta = {
        "lengths": [num_steps],
        "num_steps": num_steps,
        "num_envs": int(states.shape[0]),
        "active_joint_ids": list(env.active_joint_ids),
        "robot_uid": env.robot.uid,
        "robot_dof": int(env.robot.dof),
        "source_format": "state_action",
    }
    return {"states": replay_states, "actions": actions, "meta": meta}


def replay_trajectory(
    env: Any,
    trajectory_path: str | PathLike[str],
    mode: ReplayMode = "kinematic",
) -> None:
    """Replay a native EmbodiChain or legacy RoboSynChallenge trajectory."""
    data = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Trajectory root must be a dict, got {type(data).__name__}.")

    if {"states", "actions", "meta"}.issubset(data):
        replay_native_trajectory(env, str(trajectory_path), mode=mode)
        return

    converted = convert_state_action_trajectory(data, env.unwrapped, mode=mode)
    meta = converted["meta"]
    log_info(
        f"Replaying legacy state/action trajectory: num_envs={meta['num_envs']}, "
        f"num_steps={meta['num_steps']}, mode={mode}",
        color="green",
    )
    if mode in ("kinematic", "control"):
        log_warning(
            "This legacy file has no drawer or rigid-object states; kinematic replay "
            "reproduces robot motion only. Use --replay_mode dynamic to re-simulate "
            "object interaction from the recorded actions."
        )

    replay_env = ReplayWrapper(env.unwrapped, converted, mode=mode)
    try:
        if mode == "control":
            replay_control(replay_env)
        else:
            replay_auto(replay_env, mode)
    finally:
        replay_env.close()
