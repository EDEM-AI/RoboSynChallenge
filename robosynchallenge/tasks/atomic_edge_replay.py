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

"""Shared L1 atomic edge replay helpers for RoboSynChallenge ActionBanks."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch

from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
    get_func_tag,
    tag_edge,
    tag_node,
)
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator, ToppraPlannerCfg
from embodichain.utils import logger


__all__ = ["AtomicEdgeReplayActionBankMixin"]


class AtomicEdgeReplayActionBankMixin(ActionBank):
    """Replay RoboSyn joint edges through EmbodiChain atomic ``MoveJoints``."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AtomicEdgeReplayActionBankMixin._register_atomic_replay_edges(cls)

    @staticmethod
    def _register_atomic_replay_edges(action_bank_cls: type) -> None:
        edge_functions = get_func_tag("edge").functions.setdefault(
            action_bank_cls.__name__, {}
        )
        node_functions = get_func_tag("node").functions.setdefault(
            action_bank_cls.__name__, {}
        )

        for func_name in (
            "execute_open",
            "execute_close",
            "plan_trajectory",
            "stand_still",
        ):
            edge_functions[func_name] = getattr(
                AtomicEdgeReplayActionBankMixin, func_name
            )

        for func_name in ("execute_open", "execute_close"):
            node_functions[func_name] = getattr(
                AtomicEdgeReplayActionBankMixin, func_name
            )

    @staticmethod
    @tag_edge
    @tag_node
    def execute_open(
        env,
        return_action: bool = False,
        control_part: str | None = None,
        **kwargs: Any,
    ):
        return AtomicEdgeReplayActionBankMixin._execute_gripper_replay(
            env,
            func_name="execute_open",
            return_action=return_action,
            control_part=control_part,
            **kwargs,
        )

    @staticmethod
    @tag_edge
    @tag_node
    def execute_close(
        env,
        return_action: bool = False,
        control_part: str | None = None,
        **kwargs: Any,
    ):
        return AtomicEdgeReplayActionBankMixin._execute_gripper_replay(
            env,
            func_name="execute_close",
            return_action=return_action,
            control_part=control_part,
            **kwargs,
        )

    @staticmethod
    @tag_edge
    def plan_trajectory(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
        edge_name: str = "",
    ) -> np.ndarray:
        keyposes = [
            AtomicEdgeReplayActionBankMixin._to_qpos_tensor(
                env.affordance_datas[keypose_name], device=env.robot.device
            )
            for keypose_name in keypose_names
        ]

        if all(
            torch.linalg.norm(former - latter).sum() <= 1e-3
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            logger.log_warning(
                "Applying atomic replay plan_trajectory to very close qpos values. "
                "Using stand_still."
            )
            return AtomicEdgeReplayActionBankMixin.stand_still(
                env,
                agent_uid,
                keypose_names,
                duration,
            )

        segment_durations = AtomicEdgeReplayActionBankMixin._split_duration(
            duration, len(keyposes) - 1
        )
        action = AtomicEdgeReplayActionBankMixin._make_move_joints_action(
            env,
            control_part=agent_uid,
            sample_interval=segment_durations[0],
        )
        state = AtomicEdgeReplayActionBankMixin._make_world_state(
            env, action.joint_ids, keyposes[0]
        )

        local_segments = []
        for target_qpos, segment_duration in zip(keyposes[1:], segment_durations):
            action.cfg.sample_interval = segment_duration
            result = action.execute(
                AtomicEdgeReplayActionBankMixin._joint_position_target(target_qpos),
                state,
            )
            if not result.success:
                logger.log_warning(
                    f"Atomic MoveJoints failed for edge '{edge_name}' on {agent_uid}. "
                    "Using stand_still fallback."
                )
                return AtomicEdgeReplayActionBankMixin.stand_still(
                    env,
                    agent_uid,
                    keypose_names,
                    duration,
                )
            local_segments.append(result.trajectory[:, :, action.joint_ids])
            state = result.next_state

        return torch.cat(local_segments, dim=1)[0].detach().cpu().numpy().T

    @staticmethod
    @tag_edge
    def stand_still(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
    ) -> np.ndarray:
        start_qpos = AtomicEdgeReplayActionBankMixin._to_qpos_tensor(
            env.affordance_datas[keypose_names[0]], device=env.robot.device
        )
        action = AtomicEdgeReplayActionBankMixin._make_move_joints_action(
            env,
            control_part=agent_uid,
            sample_interval=max(int(duration), 1),
        )
        state = AtomicEdgeReplayActionBankMixin._make_world_state(
            env, action.joint_ids, start_qpos
        )
        result = action.execute(
            AtomicEdgeReplayActionBankMixin._joint_position_target(start_qpos), state
        )
        if not result.success:
            logger.log_warning(
                f"Atomic MoveJoints failed for stand_still on {agent_uid}. "
                "Using repeated qpos fallback."
            )
            return start_qpos.unsqueeze(1).repeat(1, max(int(duration), 1)).cpu().numpy()
        return result.trajectory[0, :, action.joint_ids].detach().cpu().numpy().T

    @staticmethod
    def _execute_gripper_replay(
        env,
        *,
        func_name: str,
        return_action: bool,
        control_part: str | None,
        **kwargs: Any,
    ):
        if not return_action:
            return True
        if control_part is None:
            logger.log_error(
                f"Atomic replay {func_name} requires explicit control_part in action_config.",
                ValueError,
            )

        duration = max(int(kwargs.get("duration", 1)), 1)
        original_kwargs = dict(kwargs)
        original_kwargs.pop("duration", None)
        original_qpos = AtomicEdgeReplayActionBankMixin._call_original_gripper_func(
            env,
            func_name=func_name,
            control_part=control_part,
            duration=duration,
            **original_kwargs,
        )
        if original_qpos.ndim == 1:
            original_qpos = original_qpos.reshape(1, -1)
        if original_qpos.ndim != 2:
            logger.log_error(
                f"{func_name} must produce a 2D gripper trajectory, "
                f"but got shape {original_qpos.shape}.",
                ValueError,
            )

        waypoints = torch.as_tensor(
            original_qpos.T,
            dtype=torch.float32,
            device=env.robot.device,
        )
        if waypoints.ndim == 1:
            waypoints = waypoints.unsqueeze(-1)
        if waypoints.shape[0] == 0:
            logger.log_error(f"{func_name} produced an empty gripper trajectory.")

        action = AtomicEdgeReplayActionBankMixin._make_move_joints_action(
            env,
            control_part=control_part,
            sample_interval=max(waypoints.shape[0], 1),
        )
        waypoints = torch.stack(
            [
                AtomicEdgeReplayActionBankMixin._match_qpos_dim(
                    waypoint, len(action.joint_ids)
                )
                for waypoint in waypoints
            ],
            dim=0,
        )
        state = AtomicEdgeReplayActionBankMixin._make_world_state(
            env, action.joint_ids, waypoints[0]
        )

        local_segments = [
            waypoints[:1].unsqueeze(0).repeat(action.n_envs, 1, 1)
        ]
        for target_qpos in waypoints[1:]:
            action.cfg.sample_interval = 2
            result = action.execute(
                AtomicEdgeReplayActionBankMixin._joint_position_target(target_qpos),
                state,
            )
            if not result.success:
                logger.log_warning(
                    f"Atomic MoveJoints failed for {control_part} {func_name}. "
                    "Using original gripper qpos."
                )
                return original_qpos
            local_segments.append(result.trajectory[:, -1:, action.joint_ids])
            state = result.next_state

        local_traj = torch.cat(local_segments, dim=1)
        return local_traj[0, :, : original_qpos.shape[0]].detach().cpu().numpy().T

    @staticmethod
    def _call_original_gripper_func(
        env,
        *,
        func_name: str,
        control_part: str,
        duration: int,
        **kwargs: Any,
    ) -> np.ndarray:
        action_bank_cls = env.action_bank.__class__
        for cls in action_bank_cls.mro():
            if cls is AtomicEdgeReplayActionBankMixin:
                continue
            if not issubclass(cls, ActionBank):
                continue
            func = cls.__dict__.get(func_name)
            if func is None:
                continue
            if isinstance(func, staticmethod):
                func = func.__func__
            ret = func(
                env,
                return_action=True,
                control_part=control_part,
                duration=duration,
                **kwargs,
            )
            if isinstance(ret, torch.Tensor):
                ret = ret.detach().cpu().numpy()
            return np.asarray(ret, dtype=np.float32)

        logger.log_error(
            f"Cannot find original RoboSyn {func_name} implementation for "
            f"{action_bank_cls.__name__}.",
            RuntimeError,
        )
        raise AssertionError("unreachable")

    @staticmethod
    def _make_move_joints_action(env, *, control_part: str, sample_interval: int):
        MoveJoints, MoveJointsCfg = AtomicEdgeReplayActionBankMixin._move_joints_types()
        return MoveJoints(
            AtomicEdgeReplayActionBankMixin._get_motion_generator(env),
            MoveJointsCfg(
                name=f"atomic_replay_{control_part}",
                control_part=control_part,
                sample_interval=max(int(sample_interval), 1),
            ),
        )

    @staticmethod
    def _get_motion_generator(env) -> MotionGenerator:
        attr_name = "_robosyn_atomic_edge_replay_motion_generator"
        motion_generator = getattr(env, attr_name, None)
        if motion_generator is None:
            motion_generator = MotionGenerator(
                cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
            )
            setattr(env, attr_name, motion_generator)
        return motion_generator

    @staticmethod
    def _make_world_state(env, joint_ids, start_qpos: torch.Tensor):
        WorldState = AtomicEdgeReplayActionBankMixin._world_state_type()
        full_qpos = env.robot.get_qpos().clone().to(
            device=env.robot.device, dtype=torch.float32
        )
        if full_qpos.ndim == 1:
            full_qpos = full_qpos.unsqueeze(0)
        start_qpos = AtomicEdgeReplayActionBankMixin._match_qpos_dim(
            start_qpos, len(joint_ids)
        )
        full_qpos[:, joint_ids] = start_qpos.unsqueeze(0).repeat(full_qpos.shape[0], 1)
        return WorldState(last_qpos=full_qpos)

    @staticmethod
    def _to_qpos_tensor(qpos: Any, *, device: torch.device) -> torch.Tensor:
        qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=device)
        if qpos_tensor.ndim > 1:
            qpos_tensor = qpos_tensor.reshape(-1, qpos_tensor.shape[-1])[0]
        return qpos_tensor.reshape(-1)

    @staticmethod
    def _match_qpos_dim(qpos: torch.Tensor, dim: int) -> torch.Tensor:
        qpos = qpos.reshape(-1)
        if qpos.shape[-1] == dim:
            return qpos
        if qpos.shape[-1] == 1:
            return qpos.repeat(dim)
        logger.log_error(
            f"Atomic replay qpos dim {qpos.shape[-1]} does not match control dim {dim}.",
            ValueError,
        )
        raise AssertionError("unreachable")

    @staticmethod
    def _split_duration(duration: int, n_segments: int) -> list[int]:
        if n_segments < 1:
            logger.log_error("Atomic replay requires at least one trajectory segment.")
        duration = max(int(duration), 1)
        base_duration = duration // n_segments
        remainder = duration % n_segments
        segment_durations = [
            base_duration + (1 if idx < remainder else 0)
            for idx in range(n_segments)
        ]
        if any(segment_duration < 1 for segment_duration in segment_durations):
            logger.log_error(
                f"Cannot split duration {duration} over {n_segments} segments."
            )
        return segment_durations

    @staticmethod
    def _move_joints_types():
        from embodichain.lab.sim.atomic_actions.actions import MoveJoints, MoveJointsCfg

        return MoveJoints, MoveJointsCfg

    @staticmethod
    def _joint_position_target(qpos: torch.Tensor):
        from embodichain.lab.sim.atomic_actions.core import JointPositionTarget

        return JointPositionTarget(qpos=qpos)

    @staticmethod
    def _world_state_type():
        from embodichain.lab.sim.atomic_actions.core import WorldState

        return WorldState
