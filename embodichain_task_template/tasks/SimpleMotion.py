# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
from typing import Sequence

from embodichain.lab.sim.types import EnvAction
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums
from embodichain.lab.sim.planners import (
    MotionGenerator,
    MotionGenCfg,
    MotionGenOptions,
    ToppraPlannerCfg,
    ToppraPlanOptions,
    TrajectorySampleMethod,
    PlanState,
    MoveType,
)


@register_env("SimpleMotion-v1")
class SimpleMotionEnv(EmbodiedEnv):
    """
    A simple motion generation task for testing and template purposes.

    This is a basic demonstration task that shows how to construct an environment
    and create a sequence of demonstration actions.
    """

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        self.motion_gen = MotionGenerator(
            cfg=MotionGenCfg(
                planner_cfg=ToppraPlannerCfg(
                    robot_uid=self.robot.uid,
                )
            )
        )

    def create_demo_action_list(self, *args, **kwargs) -> Sequence[EnvAction] | None:
        """Create a list of demonstration actions.

        This method generates a simple sequence of actions that can be used
        to demonstrate the task or serve as a baseline trajectory.

        Returns:
            List of action dictionaries, each containing action parameters.
        """
        qpos_fk = torch.tensor(
            [[0.0, torch.pi / 4, -torch.pi / 4, 0.0, torch.pi / 4, 0.0]],
            dtype=torch.float32,
        )
        xpos_begin = self.robot.compute_fk(
            name="left_arm", qpos=qpos_fk, to_matrix=True
        )[0]
        xpos_mid = xpos_begin.clone()
        xpos_mid[2, 3] -= 0.1  # Move down by 0.1m in Z direction
        xpos_final = xpos_mid.clone()
        xpos_final[0, 3] += 0.2  # Move forward by 0.2m in X direction

        # Generate cartesian space trajectory for left arm.
        # Currently, MotionGenerator only supports generating trajectory for one robot.
        options = MotionGenOptions(
            control_part="left_arm",
            start_qpos=self.robot.get_qpos("left_arm")[0],
            is_interpolate=True,
            is_linear=True,
            plan_opts=ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=50,
            ),
        )

        left_target_states = [
            PlanState(move_type=MoveType.EEF_MOVE, xpos=xpos)
            for xpos in [xpos_begin, xpos_mid, xpos_final]
        ]
        left_plan_result = self.motion_gen.generate(
            target_states=left_target_states, options=options
        )

        # Generate joint space trajectory for right arm.
        xpos_begin = self.robot.compute_fk(
            name="right_arm", qpos=qpos_fk, to_matrix=True
        )[0]
        xpos_mid = xpos_begin.clone()
        xpos_mid[2, 3] -= 0.1  # Move down by 0.1m in Z direction
        xpos_final = xpos_mid.clone()
        xpos_final[0, 3] += 0.2  # Move forward by 0.2m in X direction

        # IK failed, return None to indicate no demo actions can be created
        _, qpos_begin = self.robot.compute_ik(pose=xpos_begin, name="right_arm")
        _, qpos_mid = self.robot.compute_ik(pose=xpos_mid, name="right_arm")
        _, qpos_final = self.robot.compute_ik(pose=xpos_final, name="right_arm")

        options = MotionGenOptions(
            control_part="right_arm",
            is_interpolate=True,
            start_qpos=self.robot.get_qpos("right_arm")[0],
            plan_opts=ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=50,
            ),
        )

        right_target_states = [
            PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos)
            for qpos in [qpos_begin[0], qpos_mid[0], qpos_final[0]]
        ]
        right_plan_result = self.motion_gen.generate(
            target_states=right_target_states, options=options
        )

        total_len = max(
            len(left_plan_result.positions), len(right_plan_result.positions)
        )
        trajectory = torch.zeros(
            (total_len, self.robot.dof),
            dtype=torch.float32,
            device=self.device,
        )

        left_joint_ids = self.robot.get_joint_ids("left_arm")
        right_joint_ids = self.robot.get_joint_ids("right_arm")
        trajectory[:, left_joint_ids] = left_plan_result.positions
        trajectory[:, right_joint_ids] = right_plan_result.positions

        # Generate eef close to open for last 20 steps
        left_eef_ids = self.robot.get_joint_ids("left_eef")
        right_eef_ids = self.robot.get_joint_ids("right_eef")
        eef_open = self.robot.get_qpos_limits(name="left_eef")[:, :, 1]
        eef_close = self.robot.get_qpos_limits(name="left_eef")[:, :, 0]

        close_to_open = torch.stack([eef_close, eef_open], dim=1)
        close_to_open = interpolate_with_nums(
            close_to_open, interp_nums=[19], device=self.device
        )
        close_to_open = close_to_open.squeeze_(0)

        trajectory[-20:, left_eef_ids] = close_to_open
        trajectory[-20:, right_eef_ids] = close_to_open

        return trajectory[:, self.active_joint_ids]
