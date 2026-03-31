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


@register_env("OpenDrawer-v1")
class OpenDrawerEnv(EmbodiedEnv):
    """
    Environment for the task of opening a drawer. The environment includes a robot and a sliding box drawer.
    The robot is expected to learn how to open the drawer by generating a sequence of actions that move its end effectors to grasp the
    drawer handle and pull it open.
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

        self.eef_open = self.robot.get_qpos_limits(name="right_eef")[:, :, 1]
        self.eef_close = self.robot.get_qpos_limits(name="right_eef")[:, :, 0]

    def _generate_eef_motion(
        self, num_steps: int = 10, open: bool = True
    ) -> torch.Tensor:
        """Generate a simple end-effector motion for opening or closing the drawer.

        Args:
            num_steps: Number of steps in the generated trajectory.
            open: Whether to generate an opening motion (True) or closing motion (False).
        Returns:
            A tensor of shape (num_steps, dof) representing the joint positions for the end-effector motion.
        """

        if open:
            current_qpos = self.eef_close
            target_qpos = self.eef_open
        else:
            current_qpos = self.eef_open
            target_qpos = self.eef_close

        trajectory = interpolate_with_nums(
            torch.stack([current_qpos, target_qpos], dim=1),
            interp_nums=[num_steps - 1],
            device=self.device,
        ).squeeze(0)

        return trajectory

    def create_demo_action_list(self, *args, **kwargs) -> Sequence[EnvAction] | None:
        """Create a list of demonstration actions.

        This method generates a simple sequence of actions that can be used
        to demonstrate the task or serve as a baseline trajectory.

        The motion is split into four phases:
        1. Move to the starting joint configuration.
        2. Approach the drawer handle (EEF open) — ``to_handle``.
        3. Close the gripper to grasp the handle — EEF close motion.
        4. Pull the drawer open and retract — ``leave_handle``.

        Returns:
            List of action dictionaries, each containing action parameters.
        """

        # A trial starting joint configuration.
        qpos_start = torch.tensor(
            [[0.0, 2.06, -0.75, 0.0, -1.20, 1.6]],
            dtype=torch.float32,
        )

        # Generate a trajectory to the starting configuration in joint space.
        options_to_start = MotionGenOptions(
            control_part="right_arm",
            is_interpolate=True,
            start_qpos=self.robot.get_qpos("right_arm")[0],
            plan_opts=ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=50,
            ),
        )
        right_target_states = [
            PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos_start[0])
        ]
        plan_result_to_start = self.motion_gen.generate(
            target_states=right_target_states, options=options_to_start
        )

        # Generate cartesian space waypoints
        # You may use get_link_pose from articulation with specified link name,
        # and set the target pose relative to the handle pose for motion generation.
        xpos_begin = self.robot.compute_fk(
            name="right_arm", qpos=qpos_start, to_matrix=True
        )[0]
        xpos_mid = xpos_begin.clone()
        xpos_mid[0, 3] += 0.11  # Move forward by 0.1m in X direction

        # Phase 2: Approach the handle (to_handle)
        options_to_handle = MotionGenOptions(
            control_part="right_arm",
            is_interpolate=True,
            is_linear=True,
            start_qpos=qpos_start[0],
            plan_opts=ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=50,
            ),
        )
        to_handle_target_states = [
            PlanState(move_type=MoveType.EEF_MOVE, xpos=xpos)
            for xpos in [xpos_begin, xpos_mid]
        ]
        plan_result_to_handle = self.motion_gen.generate(
            target_states=to_handle_target_states, options=options_to_handle
        )

        # Phase 4: Pull drawer open and retract (leave_handle)
        # Start from where to_handle ended (xpos_mid)
        options_leave_handle = MotionGenOptions(
            control_part="right_arm",
            is_interpolate=True,
            is_linear=True,
            start_qpos=plan_result_to_handle.positions[-1],
            plan_opts=ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=50,
            ),
        )
        leave_handle_target_states = [
            PlanState(move_type=MoveType.EEF_MOVE, xpos=xpos)
            for xpos in [xpos_mid, xpos_begin]
        ]
        plan_result_leave_handle = self.motion_gen.generate(
            target_states=leave_handle_target_states, options=options_leave_handle
        )

        # Phase 3: EEF close motion (grasp the handle) — inserted between
        # to_handle and leave_handle
        num_grasp_steps = 20
        eef_grasp_motion = self._generate_eef_motion(
            num_steps=num_grasp_steps, open=False
        )

        # Compute total trajectory length
        len_to_start = len(plan_result_to_start.positions)
        len_to_handle = len(plan_result_to_handle.positions)
        len_grasp = num_grasp_steps
        len_leave_handle = len(plan_result_leave_handle.positions)
        total_len = len_to_start + len_to_handle + len_grasp + len_leave_handle

        trajectory = torch.zeros(
            (total_len, self.robot.dof),
            dtype=torch.float32,
            device=self.device,
        )

        right_joint_ids = self.robot.get_joint_ids("right_arm")
        right_eef_ids = self.robot.get_joint_ids("right_eef")

        idx = 0

        # --- Phase 1: Move to start (arm moves, EEF opens) ---
        trajectory[idx : idx + len_to_start, right_joint_ids] = (
            plan_result_to_start.positions
        )
        eef_open_motion = self._generate_eef_motion(num_steps=len_to_start, open=True)
        trajectory[idx : idx + len_to_start, right_eef_ids] = eef_open_motion
        idx += len_to_start

        # --- Phase 2: Approach handle (arm moves, EEF stays open) ---
        trajectory[idx : idx + len_to_handle, right_joint_ids] = (
            plan_result_to_handle.positions
        )
        # Keep EEF open while approaching
        trajectory[idx : idx + len_to_handle, right_eef_ids] = self.eef_open.expand(
            len_to_handle, -1
        )
        idx += len_to_handle

        # --- Phase 3: Grasp the handle (arm holds, EEF closes) ---
        # Hold the arm at the last to_handle position while closing the gripper
        trajectory[idx : idx + len_grasp, right_joint_ids] = (
            plan_result_to_handle.positions[-1].unsqueeze(0).expand(len_grasp, -1)
        )
        trajectory[idx : idx + len_grasp, right_eef_ids] = eef_grasp_motion
        idx += len_grasp

        # --- Phase 4: Pull drawer open / retract (arm moves, EEF stays closed) ---
        trajectory[idx : idx + len_leave_handle, right_joint_ids] = (
            plan_result_leave_handle.positions
        )
        # Keep EEF closed while pulling
        trajectory[idx : idx + len_leave_handle, right_eef_ids] = self.eef_close.expand(
            len_leave_handle, -1
        )
        idx += len_leave_handle

        return trajectory[:, self.active_joint_ids]
