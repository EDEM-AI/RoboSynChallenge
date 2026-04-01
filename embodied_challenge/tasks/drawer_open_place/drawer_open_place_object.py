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
from embodichain.utils.logger import log_warning


@register_env("DrawerOpenPlaceObject-v1")
class DrawerOpenPlaceObjectEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        self._drawer_link_debug_logged = False
        self.motion_gen = MotionGenerator(
            cfg=MotionGenCfg(
                planner_cfg=ToppraPlannerCfg(
                    robot_uid=self.robot.uid,
                )
            )
        )
        self.left_open_eef = self.robot.get_qpos_limits(name="left_eef")[:, :, 1]
        self.left_close_eef = self.robot.get_qpos_limits(name="left_eef")[:, :, 0]
        self.eef_open = self.robot.get_qpos_limits(name="right_eef")[:, :, 1]
        self.eef_close = self.robot.get_qpos_limits(name="right_eef")[:, :, 0]

    def _fk_pose(self, control_part: str, qpos: torch.Tensor) -> torch.Tensor:
        return self.robot.compute_fk(
            name=control_part, qpos=qpos.unsqueeze(0), to_matrix=True
        )[0]

    def _make_pose(
        self, reference_pose: torch.Tensor, xyz: Sequence[float]
    ) -> torch.Tensor:
        pose = reference_pose.clone()
        pose[:3, 3] = torch.tensor(xyz, dtype=pose.dtype, device=pose.device)
        return pose

    def _make_pose_from_axes(
        self,
        xyz: Sequence[float] | torch.Tensor,
        approach_axis: torch.Tensor,
        up_axis: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(xyz):
            xyz = torch.tensor(
                xyz, dtype=approach_axis.dtype, device=approach_axis.device
            )
        z_axis = self._normalize(approach_axis, [1.0, 0.0, 0.0])
        up_axis = up_axis - torch.dot(up_axis, z_axis) * z_axis
        y_axis = self._normalize(up_axis, [0.0, 0.0, 1.0])
        x_axis = self._normalize(
            torch.cross(y_axis, z_axis, dim=0), [0.0, 1.0, 0.0]
        )
        y_axis = self._normalize(
            torch.cross(z_axis, x_axis, dim=0), [0.0, 0.0, 1.0]
        )
        pose = torch.eye(
            4, dtype=approach_axis.dtype, device=approach_axis.device
        )
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = xyz
        return pose

    def _make_local_rotation(
        self, axis: str, angle: float, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        angle_tensor = torch.tensor(angle, dtype=dtype, device=device)
        c = torch.cos(angle_tensor)
        s = torch.sin(angle_tensor)
        rotation = torch.eye(3, dtype=dtype, device=device)
        if axis == "x":
            rotation[1, 1] = c
            rotation[1, 2] = -s
            rotation[2, 1] = s
            rotation[2, 2] = c
            return rotation
        if axis == "y":
            rotation[0, 0] = c
            rotation[0, 2] = s
            rotation[2, 0] = -s
            rotation[2, 2] = c
            return rotation
        if axis == "z":
            rotation[0, 0] = c
            rotation[0, 1] = -s
            rotation[1, 0] = s
            rotation[1, 1] = c
            return rotation
        raise ValueError(f"Unsupported rotation axis: {axis}")

    def _rotate_pose_local(
        self, reference_pose: torch.Tensor, axis: str, angle: float
    ) -> torch.Tensor:
        pose = reference_pose.clone()
        local_rotation = self._make_local_rotation(
            axis=axis,
            angle=angle,
            dtype=pose.dtype,
            device=pose.device,
        )
        pose[:3, :3] = pose[:3, :3] @ local_rotation
        return pose

    def _normalize(
        self, vector: torch.Tensor, fallback: Sequence[float]
    ) -> torch.Tensor:
        norm = torch.linalg.norm(vector)
        if norm < 1e-6:
            return torch.tensor(fallback, dtype=vector.dtype, device=vector.device)
        return vector / norm

    def _repeat_state(self, state: torch.Tensor, num_steps: int) -> torch.Tensor:
        return state.expand(num_steps, -1)

    def _interpolate_state(
        self, start: torch.Tensor, end: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        if num_steps <= 1:
            return end
        states = torch.stack([start, end], dim=1)
        interpolated = interpolate_with_nums(
            states, interp_nums=[num_steps - 1], device=self.device
        )
        return interpolated.squeeze(0)

    def _generate_eef_motion(
        self, current_qpos: torch.Tensor, target_qpos: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        return self._interpolate_state(current_qpos, target_qpos, num_steps)

    def _generate_right_eef_motion(
        self, num_steps: int = 10, open: bool = True
    ) -> torch.Tensor:
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

    def _resolve_link_name(
        self, articulation, candidate_names: Sequence[str]
    ) -> str | None:
        available_names = [str(name) for name in articulation.link_names]
        lower_to_original = {name.lower(): name for name in available_names}
        for link_name in candidate_names:
            if link_name in available_names:
                return link_name
            lower_name = link_name.lower()
            if lower_name in lower_to_original:
                return lower_to_original[lower_name]
        for candidate in candidate_names:
            lower_candidate = candidate.lower()
            for available_name in available_names:
                if lower_candidate in available_name.lower():
                    return available_name
        if not self._drawer_link_debug_logged:
            self._drawer_link_debug_logged = True
            log_warning(f"Drawer link names: {available_names}")
        return None

    def _get_link_pose(
        self, articulation, candidate_names: Sequence[str]
    ) -> torch.Tensor | None:
        link_name = self._resolve_link_name(articulation, candidate_names)
        if link_name is None:
            return None
        return articulation.get_link_pose(link_name, to_matrix=True)[0]

    def _solve_ik(
        self,
        control_part: str,
        pose: torch.Tensor,
        joint_seed: torch.Tensor,
        stage_name: str | None = None,
    ) -> torch.Tensor | None:
        result = self.robot.compute_ik(
            pose=pose.unsqueeze(0),
            joint_seed=joint_seed.unsqueeze(0),
            name=control_part,
        )
        if result is None:
            if stage_name:
                log_warning(f"IK failed at stage: {stage_name}")
            return None
        success, qpos = result
        if not bool(success[0].item()):
            if stage_name:
                log_warning(f"IK failed at stage: {stage_name}")
            return None
        return qpos[0]

    def _solve_ik_candidates(
        self,
        control_part: str,
        poses: Sequence[torch.Tensor],
        joint_seeds: Sequence[torch.Tensor],
        stage_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        preferred_seed = joint_seeds[0]
        best_result = None
        for pose_idx, pose in enumerate(poses):
            for seed_idx, joint_seed in enumerate(joint_seeds):
                qpos = self._solve_ik(control_part, pose, joint_seed)
                if qpos is not None:
                    continuity_cost = torch.linalg.norm(
                        qpos - preferred_seed
                    ).item()
                    seed_cost = torch.linalg.norm(qpos - joint_seed).item()
                    score = (
                        seed_idx * 100.0
                        + continuity_cost
                        + 0.1 * seed_cost
                        + pose_idx * 1e-3
                    )
                    if best_result is None or score < best_result[0]:
                        best_result = (score, pose, qpos)
        if best_result is not None:
            return best_result[1], best_result[2]
        log_warning(f"IK failed at stage: {stage_name}")
        return None

    def _plan_joint_motion(
        self,
        control_part: str,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        sample_interval: int = 50,
    ):
        return self.motion_gen.generate(
            target_states=[
                PlanState(move_type=MoveType.JOINT_MOVE, qpos=target_qpos)
            ],
            options=MotionGenOptions(
                control_part=control_part,
                is_interpolate=True,
                start_qpos=start_qpos,
                plan_opts=ToppraPlanOptions(
                    sample_method=TrajectorySampleMethod.QUANTITY,
                    sample_interval=sample_interval,
                ),
            ),
        )

    def _plan_linear_eef_motion(
        self,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: Sequence[torch.Tensor],
        sample_interval: int = 50,
    ):
        return self.motion_gen.generate(
            target_states=[
                PlanState(move_type=MoveType.EEF_MOVE, xpos=pose)
                for pose in target_poses
            ],
            options=MotionGenOptions(
                control_part=control_part,
                is_interpolate=True,
                is_linear=True,
                start_qpos=start_qpos,
                plan_opts=ToppraPlanOptions(
                    sample_method=TrajectorySampleMethod.QUANTITY,
                    sample_interval=sample_interval,
                ),
            ),
        )

    def _require_plan(self, plan_result, stage_name: str):
        if plan_result is None or plan_result.positions is None:
            log_warning(f"Planner returned empty result at stage: {stage_name}")
            return None
        if len(plan_result.positions) == 0:
            log_warning(f"Planner returned zero-length trajectory at stage: {stage_name}")
            return None
        success = plan_result.success
        if isinstance(success, torch.Tensor):
            success = bool(success.all().item())
        else:
            success = bool(success)
        if not success:
            log_warning(f"Planner failed at stage: {stage_name}")
            return None
        return plan_result

    def create_demo_action_list(self, *args, **kwargs) -> Sequence[EnvAction] | None:
        drawer = self.sim.get_articulation("drawer")
        duck = self.sim.get_rigid_object("duck")

        duck_pos = duck.get_local_pose(to_matrix=True)[0][:3, 3]

        left_home_qpos = self.robot.get_qpos("left_arm")[0]
        right_home_qpos = self.robot.get_qpos("right_arm")[0]
        left_init_eef = self.robot.get_qpos("left_eef")
        right_init_eef = self.robot.get_qpos("right_eef")

        # Right arm initial target pose for grasping the drawer handle
        # Calculate Forward Kinematics (FK) for the base pose, add offsets, and compute Inverse Kinematics (IK)
        right_qpos_start_base = torch.tensor(
            [0.0, 2.06, -0.75, 0.0, -1.20, 1.6],
            dtype=torch.float32,
            device=self.device,
        )
        right_reference_pose_base = self._fk_pose("right_arm", right_qpos_start_base)
        
        # ==========================================
        # Right Arm: Adapt to the drawer's current position (x=0.88, y=-0.10)
        # ==========================================
        # Offsets are relative to the base configuration (drawer x=0.90, y=0.10)
        right_reference_pose = right_reference_pose_base.clone()
        right_reference_pose[0, 3] -= 0.02 # x offset (-0.02m)
        right_reference_pose[1, 3] += 0.20 # y offset (+0.20m)
        
        # Compute IK to get the adapted joint angles
        _, right_qpos_start = self.robot.compute_ik(pose=right_reference_pose.unsqueeze(0), name="right_arm")
        right_qpos_start = right_qpos_start[0]

        plan_result_to_start = self._require_plan(
            self._plan_joint_motion("right_arm", right_home_qpos, right_qpos_start),
            "right_to_start",
        )
        if plan_result_to_start is None:
            return None

        right_xpos_begin = right_reference_pose.clone()
        right_xpos_mid = right_xpos_begin.clone()
        right_xpos_mid[0, 3] += 0.06
        right_xpos_pull = right_xpos_begin.clone()
        # Reduce pull distance to avoid excessive stretch and jitter
        right_xpos_pull[0, 3] -= 0.11

        plan_result_to_handle = self._require_plan(
            self._plan_linear_eef_motion(
                "right_arm",
                right_qpos_start,
                [right_xpos_begin, right_xpos_mid],
            ),
            "right_to_handle",
        )
        if plan_result_to_handle is None:
            return None

        plan_result_leave_handle = self._require_plan(
            self._plan_linear_eef_motion(
                "right_arm",
                plan_result_to_handle.positions[-1],
                [right_xpos_mid, right_xpos_begin, right_xpos_pull],
            ),
            "right_leave_handle",
        )
        if plan_result_leave_handle is None:
            return None

        # ==========================================
        # Left Arm Duck Grasping
        # ==========================================
        
        # Calculate duck target position
        ox, oy, oz = (
            float(duck_pos[0]) - 0.015,  # Offset to avoid collision with duck body
            float(duck_pos[1]) + 0.010,  # Offset to align with the thin neck
            float(duck_pos[2]),
        )
        # Safe z-height for moving duck across the drawer
        safe_z = oz + 0.22
        left_pregrasp_z = oz + 0.10  # Pre-grasp height (10cm above duck)
        left_grasp_z = oz + 0.04     # Grasp height (aligned with duck neck/body)

        # Left arm initial target pose for grasping the duck
        left_seed_qpos = torch.tensor(
            [0.0, torch.pi / 4, -torch.pi / 4, 0.0, torch.pi / 4, 0.0],
            dtype=torch.float32,
            device=self.device,
        )

        # Use the exact forward kinematics pose of the seed configuration as a reference
        left_reference_pose = self._fk_pose("left_arm", left_seed_qpos)
        
        # Add slight variations to the target pose to improve IK success rate
        left_hover_candidates = [
            self._make_pose(left_reference_pose, [ox + dx, oy + dy, safe_z])
            for dx in (0.0, 0.002, -0.002) for dy in (0.0, 0.002, -0.002)
        ]
        left_start_candidates = [
            self._make_pose(left_reference_pose, [ox + dx, oy + dy, left_pregrasp_z])
            for dx in (0.0, 0.002, -0.002) for dy in (0.0, 0.002, -0.002)
        ]
        left_grasp_candidates = [
            self._make_pose(left_reference_pose, [ox + dx, oy + dy, left_grasp_z])
            for dx in (0.0, 0.002, -0.002) for dy in (0.0, 0.002, -0.002)
        ]

        # Solve IK for left arm and select the best smooth poses
        # Solve hover pose
        left_hover_result = self._solve_ik_candidates(
            "left_arm", left_hover_candidates, [left_seed_qpos, left_home_qpos], "left_hover"
        )
        if left_hover_result is None:
            return None
        _, left_hover_qpos = left_hover_result
        left_hover_pose = self._fk_pose("left_arm", left_hover_qpos)

        # Solve pre-grasp pose
        left_start_result = self._solve_ik_candidates(
            "left_arm", left_start_candidates, [left_hover_qpos, left_seed_qpos], "left_start"
        )
        if left_start_result is None:
            return None
        _, left_start_qpos = left_start_result
        left_start_pose = self._fk_pose("left_arm", left_start_qpos)

        # Solve grasp pose
        left_grasp_result = self._solve_ik_candidates(
            "left_arm", left_grasp_candidates, [left_start_qpos, left_hover_qpos], "left_grasp"
        )
        if left_grasp_result is None:
            return None
        _, left_grasp_qpos = left_grasp_result
        left_grasp_actual_pose = self._fk_pose("left_arm", left_grasp_qpos)
        
        # Lift pose reuses the exact grasp pose, only changing Z height
        left_lift_pose = left_grasp_actual_pose.clone()
        left_lift_pose[2, 3] = safe_z
        
        # Plan joint motion from Home to Hover
        left_plan_to_hover = self._require_plan(
            self._plan_joint_motion(
                "left_arm", left_home_qpos, left_hover_qpos
            ),
            "left_to_hover"
        )
        if left_plan_to_hover is None:
            return None

        # Plan Cartesian linear motion from Hover -> Pregrasp -> Grasp
        left_plan_to_object = self._require_plan(
            self._plan_linear_eef_motion(
                "left_arm",
                left_hover_qpos,
                [left_hover_pose, left_start_pose, left_grasp_actual_pose],
            ),
            "left_to_object"
        )
        if left_plan_to_object is None:
            return None

        left_plan_lift = self._require_plan(
            self._plan_linear_eef_motion(
                "left_arm",
                left_plan_to_object.positions[-1],
                [left_grasp_actual_pose, left_lift_pose],
            ),
            "left_lift"
        )
        if left_plan_lift is None:
            return None

        # ==========================================
        # Place Duck in Drawer
        # ==========================================
        
        # Calculate candidates for hover poses above the drawer
        # Set placement coordinates based on the opened drawer's position
        place_x = 0.705
        place_y = -0.05
        place_z_hover = safe_z
        place_z_drop = 1.05 # Higher drop point to let duck fall naturally
        
        # Generate candidates for placing hover pose
        left_place_hover_candidates = []
        for dx in (0.0, 0.01, -0.01):
            for dy in (0.0, 0.01, -0.01):
                base_pose = self._make_pose(left_reference_pose, [place_x + dx, place_y + dy, place_z_hover])
                left_place_hover_candidates.append(base_pose)
        
        left_place_hover_result = self._solve_ik_candidates(
            "left_arm", left_place_hover_candidates, [left_plan_lift.positions[-1], left_home_qpos], "left_place_hover"
        )
        if left_place_hover_result is None:
            return None
        _, left_place_hover_qpos = left_place_hover_result
        left_place_hover_pose = self._fk_pose("left_arm", left_place_hover_qpos)
        
        # Calculate drop pose
        left_place_drop_pose = left_place_hover_pose.clone()
        left_place_drop_pose[2, 3] = place_z_drop
        
        # Plan joint motion to drawer hover position
        left_plan_to_place_hover = self._require_plan(
            self._plan_joint_motion(
                "left_arm", left_plan_lift.positions[-1], left_place_hover_qpos
            ),
            "left_to_place_hover"
        )
        if left_plan_to_place_hover is None:
            return None
            
        # Plan Cartesian motion to drop position
        left_plan_to_place_drop = self._require_plan(
            self._plan_linear_eef_motion(
                "left_arm",
                left_place_hover_qpos,
                [left_place_hover_pose, left_place_drop_pose],
            ),
            "left_to_place_drop"
        )
        if left_plan_to_place_drop is None:
            return None
            
        # Plan Cartesian motion to lift back up
        left_plan_place_lift = self._require_plan(
            self._plan_linear_eef_motion(
                "left_arm",
                left_plan_to_place_drop.positions[-1],
                [left_place_drop_pose, left_place_hover_pose],
            ),
            "left_place_lift"
        )
        if left_plan_place_lift is None:
            return None

        # ==========================================
        # Trajectory Lengths and Timing
        # ==========================================
        
        right_handle_grasp_steps = 20
        right_handle_settle_steps = 10
        right_release_handle_steps = 30
        right_post_release_settle_steps = 15

        right_len_to_start = len(plan_result_to_start.positions)
        right_len_to_handle = len(plan_result_to_handle.positions)
        right_len_handle_grasp = right_handle_grasp_steps
        right_len_handle_settle = right_handle_settle_steps
        right_len_leave_handle = len(plan_result_leave_handle.positions)
        right_len_release_handle = right_release_handle_steps
        right_len_post_release = right_post_release_settle_steps
        
        right_total_len = (
            right_len_to_start
            + right_len_to_handle
            + right_len_handle_grasp
            + right_len_handle_settle
            + right_len_leave_handle
            + right_len_release_handle
            + right_len_post_release
        )

        left_grasp_steps = 20
        left_settle_steps = 10
        
        left_len_to_hover = len(left_plan_to_hover.positions)
        left_len_to_object = len(left_plan_to_object.positions)
        left_len_lift = len(left_plan_lift.positions)
        
        left_len_to_place_hover = len(left_plan_to_place_hover.positions)
        left_len_to_place_drop = len(left_plan_to_place_drop.positions)
        left_release_steps = 20
        left_len_place_lift = len(left_plan_place_lift.positions)
        
        left_total_len = (
            left_len_to_hover
            + left_len_to_object
            + left_grasp_steps
            + left_settle_steps
            + left_len_lift
            + left_len_to_place_hover
            + left_len_to_place_drop
            + left_release_steps
            + left_len_place_lift
        )

        total_len = right_total_len + left_total_len
        trajectory = torch.zeros(
            (total_len, self.robot.dof),
            dtype=torch.float32,
            device=self.device,
        )

        left_joint_ids = self.robot.get_joint_ids("left_arm")
        right_joint_ids = self.robot.get_joint_ids("right_arm")
        left_eef_ids = self.robot.get_joint_ids("left_eef")
        right_eef_ids = self.robot.get_joint_ids("right_eef")

        # Default gripper state (both open)
        trajectory[:, left_eef_ids] = self.left_open_eef
        trajectory[:, right_eef_ids] = self.eef_open
        
        # Left arm remains at home while right arm moves
        trajectory[:right_total_len, left_joint_ids] = self._repeat_state(
            left_home_qpos, right_total_len
        )
        # Right arm holds its final position while left arm moves
        trajectory[right_total_len:, right_joint_ids] = self._repeat_state(
            plan_result_leave_handle.positions[-1], left_total_len
        )

        # ==========================================
        # Right Arm Trajectory Concatenation
        # ==========================================
        idx = 0
        trajectory[idx:idx + right_len_to_start, right_joint_ids] = (
            plan_result_to_start.positions
        )
        trajectory[idx:idx + right_len_to_start, right_eef_ids] = (
            self._generate_right_eef_motion(num_steps=right_len_to_start, open=True)
        )
        idx += right_len_to_start

        trajectory[idx:idx + right_len_to_handle, right_joint_ids] = (
            plan_result_to_handle.positions
        )
        trajectory[idx:idx + right_len_to_handle, right_eef_ids] = (
            self.eef_open.expand(right_len_to_handle, -1)
        )
        idx += right_len_to_handle

        trajectory[idx:idx + right_len_handle_grasp, right_joint_ids] = (
            plan_result_to_handle.positions[-1].unsqueeze(0).expand(
                right_len_handle_grasp, -1
            )
        )
        trajectory[idx:idx + right_len_handle_grasp, right_eef_ids] = (
            self._generate_right_eef_motion(
                num_steps=right_len_handle_grasp, open=False
            )
        )
        idx += right_len_handle_grasp

        trajectory[idx:idx + right_len_handle_settle, right_joint_ids] = (
            plan_result_to_handle.positions[-1].unsqueeze(0).expand(
                right_len_handle_settle, -1
            )
        )
        trajectory[idx:idx + right_len_handle_settle, right_eef_ids] = (
            self.eef_close.expand(right_len_handle_settle, -1)
        )
        idx += right_len_handle_settle

        trajectory[idx:idx + right_len_leave_handle, right_joint_ids] = (
            plan_result_leave_handle.positions
        )
        trajectory[idx:idx + right_len_leave_handle, right_eef_ids] = (
            self.eef_close.expand(right_len_leave_handle, -1)
        )
        idx += right_len_leave_handle

        trajectory[idx:idx + right_len_release_handle, right_joint_ids] = (
            plan_result_leave_handle.positions[-1].unsqueeze(0).expand(
                right_len_release_handle, -1
            )
        )
        trajectory[idx:idx + right_len_release_handle, right_eef_ids] = (
            self._generate_eef_motion(
                self.eef_close, self.eef_open, right_len_release_handle
            )
        )
        idx += right_len_release_handle

        # Add stabilization phase after release
        trajectory[idx:idx + right_len_post_release, right_joint_ids] = (
            plan_result_leave_handle.positions[-1].unsqueeze(0).expand(
                right_len_post_release, -1
            )
        )
        trajectory[idx:idx + right_len_post_release, right_eef_ids] = (
            self.eef_open.expand(right_len_post_release, -1)
        )
        idx += right_len_post_release

        # ==========================================
        # Left Arm Trajectory Concatenation
        # ==========================================
        trajectory[idx:idx + left_len_to_hover, left_joint_ids] = (
            left_plan_to_hover.positions
        )
        trajectory[idx:idx + left_len_to_hover, left_eef_ids] = (
            self.left_open_eef.expand(left_len_to_hover, -1)
        )
        idx += left_len_to_hover

        trajectory[idx:idx + left_len_to_object, left_joint_ids] = (
            left_plan_to_object.positions
        )
        trajectory[idx:idx + left_len_to_object, left_eef_ids] = (
            self.left_open_eef.expand(left_len_to_object, -1)
        )
        idx += left_len_to_object

        trajectory[idx:idx + left_grasp_steps, left_joint_ids] = (
            left_plan_to_object.positions[-1].unsqueeze(0).expand(
                left_grasp_steps, -1
            )
        )
        trajectory[idx:idx + left_grasp_steps, left_eef_ids] = (
            self._generate_eef_motion(
                self.left_open_eef, self.left_close_eef, left_grasp_steps
            )
        )
        idx += left_grasp_steps

        trajectory[idx:idx + left_settle_steps, left_joint_ids] = (
            left_plan_to_object.positions[-1].unsqueeze(0).expand(
                left_settle_steps, -1
            )
        )
        trajectory[idx:idx + left_settle_steps, left_eef_ids] = (
            self.left_close_eef.expand(left_settle_steps, -1)
        )
        idx += left_settle_steps

        trajectory[idx:idx + left_len_lift, left_joint_ids] = (
            left_plan_lift.positions
        )
        trajectory[idx:idx + left_len_lift, left_eef_ids] = (
            self.left_close_eef.expand(left_len_lift, -1)
        )
        idx += left_len_lift

        # Move to hover above the drawer
        trajectory[idx:idx + left_len_to_place_hover, left_joint_ids] = (
            left_plan_to_place_hover.positions
        )
        trajectory[idx:idx + left_len_to_place_hover, left_eef_ids] = (
            self.left_close_eef.expand(left_len_to_place_hover, -1)
        )
        idx += left_len_to_place_hover
        
        # Descend to the drop point
        trajectory[idx:idx + left_len_to_place_drop, left_joint_ids] = (
            left_plan_to_place_drop.positions
        )
        trajectory[idx:idx + left_len_to_place_drop, left_eef_ids] = (
            self.left_close_eef.expand(left_len_to_place_drop, -1)
        )
        idx += left_len_to_place_drop
        
        # Open gripper to release the duck
        trajectory[idx:idx + left_release_steps, left_joint_ids] = (
            left_plan_to_place_drop.positions[-1].unsqueeze(0).expand(
                left_release_steps, -1
            )
        )
        trajectory[idx:idx + left_release_steps, left_eef_ids] = (
            self._generate_eef_motion(
                self.left_close_eef, self.left_open_eef, left_release_steps
            )
        )
        idx += left_release_steps
        
        # Lift up to reset
        trajectory[idx:idx + left_len_place_lift, left_joint_ids] = (
            left_plan_place_lift.positions
        )
        trajectory[idx:idx + left_len_place_lift, left_eef_ids] = (
            self.left_open_eef.expand(left_len_place_lift, -1)
        )

        return trajectory[:, self.active_joint_ids]
