import numpy as np
import torch
from typing import Sequence

from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
    tag_edge,
    tag_node,
)
from embodichain.lab.gym.utils.misc import resolve_env_params
from embodichain.lab.sim.planners import (
    MotionGenerator,
    MotionGenCfg,
    MotionGenOptions,
    MoveType,
    PlanState,
    ToppraPlanOptions,
    ToppraPlannerCfg,
    TrajectorySampleMethod,
)
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums
from embodichain.utils import logger

__all__ = ["DrawerOpenPlaceActionBank"]


class DrawerOpenPlaceActionBank(ActionBank):
    @staticmethod
    def _ensure_affordance_datas(env):
        if hasattr(env, "affordance_datas") is False or env.affordance_datas is None:
            env.affordance_datas = {}
        return env.affordance_datas

    @staticmethod
    def _device(env):
        return getattr(env, "device", torch.device("cpu"))

    @staticmethod
    def _as_tensor(env, value, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=DrawerOpenPlaceActionBank._device(env), dtype=dtype)
        return torch.as_tensor(
            value, dtype=dtype, device=DrawerOpenPlaceActionBank._device(env)
        )

    @staticmethod
    def _as_numpy(env, value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _pose_matrix(env, value) -> torch.Tensor:
        pose = DrawerOpenPlaceActionBank._as_tensor(env, value)
        if pose.ndim == 3:
            return pose[0]
        if pose.ndim == 2:
            return pose
        raise ValueError(f"Unsupported pose shape: {tuple(pose.shape)}")

    @staticmethod
    def _get_motion_generator(env) -> MotionGenerator:
        motion_gen = getattr(env, "_drawer_open_place_motion_gen", None)
        if motion_gen is None:
            motion_gen = MotionGenerator(
                cfg=MotionGenCfg(
                    planner_cfg=ToppraPlannerCfg(
                        robot_uid=env.robot.uid,
                    )
                )
            )
            env._drawer_open_place_motion_gen = motion_gen
        return motion_gen

    @staticmethod
    def _normalize(env, vector: torch.Tensor, fallback: Sequence[float]) -> torch.Tensor:
        norm = torch.linalg.norm(vector)
        if norm < 1e-6:
            return torch.tensor(
                fallback, dtype=vector.dtype, device=vector.device
            )
        return vector / norm

    @staticmethod
    def _fk_pose(env, control_part: str, qpos: torch.Tensor) -> torch.Tensor:
        return env.robot.compute_fk(
            name=control_part, qpos=qpos.unsqueeze(0), to_matrix=True
        )[0]

    @staticmethod
    def _make_pose(
        env, reference_pose: torch.Tensor, xyz: Sequence[float] | torch.Tensor
    ) -> torch.Tensor:
        pose = reference_pose.clone()
        pose[:3, 3] = DrawerOpenPlaceActionBank._as_tensor(env, xyz, pose.dtype)
        return pose

    @staticmethod
    def _make_pose_from_axes(
        env,
        xyz: Sequence[float] | torch.Tensor,
        approach_axis: torch.Tensor,
        up_axis: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(xyz):
            xyz = torch.tensor(
                xyz, dtype=approach_axis.dtype, device=approach_axis.device
            )
        z_axis = DrawerOpenPlaceActionBank._normalize(env, approach_axis, [1.0, 0.0, 0.0])
        up_axis = up_axis - torch.dot(up_axis, z_axis) * z_axis
        y_axis = DrawerOpenPlaceActionBank._normalize(env, up_axis, [0.0, 0.0, 1.0])
        x_axis = DrawerOpenPlaceActionBank._normalize(
            env, torch.cross(y_axis, z_axis, dim=0), [0.0, 1.0, 0.0]
        )
        y_axis = DrawerOpenPlaceActionBank._normalize(
            env, torch.cross(z_axis, x_axis, dim=0), [0.0, 0.0, 1.0]
        )
        pose = torch.eye(
            4, dtype=approach_axis.dtype, device=approach_axis.device
        )
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = xyz
        return pose

    @staticmethod
    def _make_local_rotation(
        axis: str, angle: float, dtype: torch.dtype, device: torch.device
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

    @staticmethod
    def _rotate_pose_local(
        pose: torch.Tensor, axis: str, angle: float
    ) -> torch.Tensor:
        ret = pose.clone()
        local_rotation = DrawerOpenPlaceActionBank._make_local_rotation(
            axis=axis,
            angle=angle,
            dtype=ret.dtype,
            device=ret.device,
        )
        ret[:3, :3] = ret[:3, :3] @ local_rotation
        return ret

    @staticmethod
    def _interpolate_state(
        env, start: torch.Tensor, end: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        start = DrawerOpenPlaceActionBank._as_tensor(env, start).reshape(-1)
        end = DrawerOpenPlaceActionBank._as_tensor(env, end).reshape(-1)
        if num_steps <= 1:
            return end.unsqueeze(0)
        states = torch.stack([start, end], dim=0).unsqueeze(0)
        interpolated = interpolate_with_nums(
            states, interp_nums=[num_steps - 1], device=DrawerOpenPlaceActionBank._device(env)
        )
        return interpolated.squeeze(0)

    @staticmethod
    def _resolve_link_name(articulation, candidate_names: Sequence[str]) -> str | None:
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
        return None

    @staticmethod
    def _get_link_pose(env, articulation, candidate_names: Sequence[str]) -> torch.Tensor | None:
        link_name = DrawerOpenPlaceActionBank._resolve_link_name(
            articulation, candidate_names
        )
        if link_name is None:
            return None
        return articulation.get_link_pose(link_name, to_matrix=True)[0]

    @staticmethod
    def _solve_ik(
        env,
        control_part: str,
        pose: torch.Tensor,
        joint_seed: torch.Tensor,
        stage_name: str,
    ) -> torch.Tensor | None:
        result = env.robot.compute_ik(
            pose=pose.unsqueeze(0),
            joint_seed=joint_seed.unsqueeze(0),
            name=control_part,
        )
        if result is None:
            logger.log_warning(f"IK failed at stage: {stage_name}")
            return None
        success, qpos = result
        if bool(success[0].item()) is False:
            logger.log_warning(f"IK failed at stage: {stage_name}")
            return None
        return qpos[0]

    @staticmethod
    def _solve_ik_candidates(
        env,
        control_part: str,
        poses: Sequence[torch.Tensor],
        joint_seeds: Sequence[torch.Tensor],
        stage_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        preferred_seed = joint_seeds[0]
        best_result = None
        for pose_idx, pose in enumerate(poses):
            for seed_idx, joint_seed in enumerate(joint_seeds):
                qpos = DrawerOpenPlaceActionBank._solve_ik(
                    env, control_part, pose, joint_seed, stage_name
                )
                if qpos is None:
                    continue
                continuity_cost = torch.linalg.norm(qpos - preferred_seed).item()
                seed_cost = torch.linalg.norm(qpos - joint_seed).item()
                score = seed_idx * 100.0 + continuity_cost + 0.1 * seed_cost + pose_idx * 1e-3
                if best_result is None or score < best_result[0]:
                    best_result = (score, pose, qpos)
        if best_result is None:
            logger.log_warning(f"IK failed at stage: {stage_name}")
            return None
        return best_result[1], best_result[2]

    @staticmethod
    def _plan_joint_motion(
        env,
        control_part: str,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        sample_interval: int,
    ):
        motion_gen = DrawerOpenPlaceActionBank._get_motion_generator(env)
        return motion_gen.generate(
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

    @staticmethod
    def _plan_linear_eef_motion(
        env,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: Sequence[torch.Tensor],
        sample_interval: int,
    ):
        motion_gen = DrawerOpenPlaceActionBank._get_motion_generator(env)
        return motion_gen.generate(
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

    @staticmethod
    def _require_plan(env, plan_result, stage_name: str):
        if plan_result is None or plan_result.positions is None:
            logger.log_warning(f"Planner returned empty result at stage: {stage_name}")
            return None
        if len(plan_result.positions) == 0:
            logger.log_warning(
                f"Planner returned zero-length trajectory at stage: {stage_name}"
            )
            return None
        success = plan_result.success
        if isinstance(success, torch.Tensor):
            success = bool(success.all().item())
        else:
            success = bool(success)
        if success is False:
            logger.log_warning(f"Planner failed at stage: {stage_name}")
            return None
        return plan_result

    @staticmethod
    def _ensure_right_arm_plan(env):
        affordance_datas = DrawerOpenPlaceActionBank._ensure_affordance_datas(env)
        if affordance_datas.get("_right_arm_drawer_plan_ready", False):
            return True

        drawer_pose = DrawerOpenPlaceActionBank._pose_matrix(
            env, affordance_datas["drawer_pose"]
        )
        right_home_qpos = DrawerOpenPlaceActionBank._as_tensor(
            env, env.robot.get_qpos("right_arm")[0]
        )

        base_qpos = torch.tensor(
            [0.0, 2.06, -0.75, 0.0, -1.20, 1.6],
            dtype=torch.float32,
            device=DrawerOpenPlaceActionBank._device(env),
        )
        nominal_drawer_xy = DrawerOpenPlaceActionBank._as_tensor(
            env, affordance_datas.get("nominal_drawer_xy", [0.90, 0.10])
        )
        drawer_pull_distance = float(affordance_datas.get("drawer_pull_distance", -0.12))

        reference_pose = DrawerOpenPlaceActionBank._fk_pose(env, "right_arm", base_qpos)
        drawer_delta_xy = drawer_pose[:2, 3] - nominal_drawer_xy
        reference_pose[0, 3] += drawer_delta_xy[0]
        reference_pose[1, 3] -= drawer_delta_xy[1]

        # Original logic: right_reference_pose is the 'begin' pose.
        # It moves +0.06 to 'mid' (contact) and then pulls to -0.12 from 'begin'.
        handle_start_pose = reference_pose.clone()
        handle_contact_pose = reference_pose.clone()
        handle_contact_pose[0, 3] += 0.06
        handle_pull_pose = reference_pose.clone()
        handle_pull_pose[0, 3] += drawer_pull_distance
        right_retreat_pose = handle_pull_pose.clone()
        right_retreat_pose[0, 3] -= 0.05

        handle_start_qpos = DrawerOpenPlaceActionBank._solve_ik(
            env,
            "right_arm",
            handle_start_pose,
            right_home_qpos,
            "right_handle_start",
        )
        if handle_start_qpos is None:
            return False

        handle_contact_qpos = DrawerOpenPlaceActionBank._solve_ik(
            env,
            "right_arm",
            handle_contact_pose,
            handle_start_qpos,
            "right_handle_contact",
        )
        if handle_contact_qpos is None:
            return False

        handle_pull_qpos = DrawerOpenPlaceActionBank._solve_ik(
            env,
            "right_arm",
            handle_pull_pose,
            handle_contact_qpos,
            "right_handle_pull",
        )
        if handle_pull_qpos is None:
            return False

        right_retreat_qpos = DrawerOpenPlaceActionBank._solve_ik(
            env,
            "right_arm",
            right_retreat_pose,
            handle_pull_qpos,
            "right_retreat",
        )
        if right_retreat_qpos is None:
            return False

        affordance_datas["right_arm_init_qpos"] = right_home_qpos
        affordance_datas["right_arm_handle_start_pose"] = handle_start_pose
        affordance_datas["right_arm_handle_start_qpos"] = handle_start_qpos
        affordance_datas["right_arm_handle_contact_pose"] = handle_contact_pose
        affordance_datas["right_arm_handle_contact_qpos"] = handle_contact_qpos
        affordance_datas["right_arm_drawer_pull_pose"] = handle_pull_pose
        affordance_datas["right_arm_drawer_pull_qpos"] = handle_pull_qpos
        affordance_datas["right_arm_retreat_pose"] = right_retreat_pose
        affordance_datas["right_arm_retreat_qpos"] = right_retreat_qpos
        affordance_datas["_right_arm_drawer_plan_ready"] = True
        return True

    @staticmethod
    def _ensure_left_arm_plan(env):
        affordance_datas = DrawerOpenPlaceActionBank._ensure_affordance_datas(env)
        if affordance_datas.get("_left_arm_pick_place_plan_ready", False):
            return True

        duck_pose = DrawerOpenPlaceActionBank._pose_matrix(
            env, affordance_datas["duck_pose"]
        )
        drawer_pose = DrawerOpenPlaceActionBank._pose_matrix(
            env, affordance_datas["drawer_pose"]
        )

        left_home_qpos = DrawerOpenPlaceActionBank._as_tensor(
            env, env.robot.get_qpos("left_arm")[0]
        )

        left_seed_qpos = torch.tensor(
            [0.0, torch.pi / 4, -torch.pi / 4, 0.0, torch.pi / 4, 0.0],
            dtype=torch.float32,
            device=DrawerOpenPlaceActionBank._device(env),
        )
        left_reference_pose = DrawerOpenPlaceActionBank._fk_pose(
            env, "left_arm", left_seed_qpos
        )

        duck_grasp_offset_xyz = DrawerOpenPlaceActionBank._as_tensor(
            env, affordance_datas.get("duck_grasp_offset_xyz", [-0.015, 0.010, 0.035])
        )
        duck_hover_height = float(affordance_datas.get("duck_hover_height", 0.24))
        duck_pregrasp_height = float(affordance_datas.get("duck_pregrasp_height", 0.10))
        drawer_pull_distance = float(affordance_datas.get("drawer_pull_distance", -0.12))
        drawer_inner_offset_xyz = DrawerOpenPlaceActionBank._as_tensor(
            env, affordance_datas.get("drawer_inner_offset_xyz", [-0.04, 0.05, 0.295])
        )
        drawer_place_hover_offset = DrawerOpenPlaceActionBank._as_tensor(
            env, affordance_datas.get("drawer_place_hover_offset_xyz", [0.0, 0.0, 0.12])
        )
        old_safe_z = 1.04

        # safe_z, hover_xyz, pregrasp_xyz, grasp_xyz calculation exactly matching original parameters
        duck_target_xyz = duck_pose[:3, 3].clone()
        ox = float(duck_target_xyz[0]) - 0.015
        oy = float(duck_target_xyz[1]) + 0.010
        oz = float(duck_target_xyz[2])
        safe_z = min(oz + duck_hover_height, old_safe_z)
        
        hover_xyz = [ox, oy, safe_z]
        pregrasp_xyz = [ox, oy, oz + duck_pregrasp_height]
        grasp_xyz = [ox, oy, oz + 0.035] # Hardcoded 0.035 grasp z in original

        left_hover_candidates = [
            DrawerOpenPlaceActionBank._make_pose(
                env,
                left_reference_pose,
                [float(hover_xyz[0] + dx), float(hover_xyz[1] + dy), float(hover_xyz[2])],
            )
            for dx in (0.0, 0.002, -0.002)
            for dy in (0.0, 0.002, -0.002)
        ]
        left_pregrasp_candidates = [
            DrawerOpenPlaceActionBank._make_pose(
                env,
                left_reference_pose,
                [
                    float(pregrasp_xyz[0] + dx),
                    float(pregrasp_xyz[1] + dy),
                    float(pregrasp_xyz[2]),
                ],
            )
            for dx in (0.0, 0.002, -0.002)
            for dy in (0.0, 0.002, -0.002)
        ]
        left_grasp_candidates = [
            DrawerOpenPlaceActionBank._make_pose(
                env,
                left_reference_pose,
                [float(grasp_xyz[0] + dx), float(grasp_xyz[1] + dy), float(grasp_xyz[2])],
            )
            for dx in (0.0, 0.002, -0.002)
            for dy in (0.0, 0.002, -0.002)
        ]

        left_hover_result = DrawerOpenPlaceActionBank._solve_ik_candidates(
            env,
            "left_arm",
            left_hover_candidates,
            [left_seed_qpos, left_home_qpos],
            "left_hover",
        )
        if left_hover_result is None:
            return False
        _, left_hover_qpos = left_hover_result
        left_hover_pose = DrawerOpenPlaceActionBank._fk_pose(
            env, "left_arm", left_hover_qpos
        )

        left_pregrasp_result = DrawerOpenPlaceActionBank._solve_ik_candidates(
            env,
            "left_arm",
            left_pregrasp_candidates,
            [left_hover_qpos, left_seed_qpos],
            "left_pregrasp",
        )
        if left_pregrasp_result is None:
            return False
        _, left_pregrasp_qpos = left_pregrasp_result
        left_pregrasp_pose = DrawerOpenPlaceActionBank._fk_pose(
            env, "left_arm", left_pregrasp_qpos
        )

        left_grasp_result = DrawerOpenPlaceActionBank._solve_ik_candidates(
            env,
            "left_arm",
            left_grasp_candidates,
            [left_pregrasp_qpos, left_hover_qpos],
            "left_grasp",
        )
        if left_grasp_result is None:
            return False
        _, left_grasp_qpos = left_grasp_result
        left_grasp_pose = DrawerOpenPlaceActionBank._fk_pose(
            env, "left_arm", left_grasp_qpos
        )

        left_lift_pose = left_grasp_pose.clone()
        left_lift_pose[2, 3] = safe_z
        left_plan_lift = DrawerOpenPlaceActionBank._require_plan(
            env,
            DrawerOpenPlaceActionBank._plan_linear_eef_motion(
                env,
                "left_arm",
                left_grasp_qpos,
                [left_grasp_pose, left_lift_pose],
                sample_interval=50,
            ),
            "left_lift",
        )
        if left_plan_lift is None:
            return False
        left_lift_qpos = left_plan_lift.positions[-1]

        # Decouple hover height from safe_z to prevent IK failures when reaching forward
        # Adjusted X to 0.680 and Y to -0.05 as per original parameters
        place_drop_xyz = torch.tensor([0.680, -0.05, 1.05], dtype=torch.float32, device=DrawerOpenPlaceActionBank._device(env))
        place_z_hover = old_safe_z
        place_hover_xyz = place_drop_xyz.clone()
        place_hover_xyz[2] = place_z_hover

        left_place_hover_candidates = [
            DrawerOpenPlaceActionBank._make_pose(
                env,
                left_reference_pose,
                [
                    float(place_hover_xyz[0] + dx),
                    float(place_hover_xyz[1] + dy),
                    float(place_hover_xyz[2]),
                ],
            )
            for dx in (0.0, 0.01, -0.01)
            for dy in (0.0, 0.01, -0.01)
        ]
        left_place_hover_candidate_xyzs = [
            DrawerOpenPlaceActionBank._as_numpy(env, pose[:3, 3]).tolist()
            for pose in left_place_hover_candidates
        ]
        logger.log_warning(
            "old left_place_hover debug | "
            f"place_xyz=({float(place_hover_xyz[0]):.4f}, {float(place_hover_xyz[1]):.4f}, {float(place_hover_xyz[2]):.4f}) | "
            f"candidate_xyzs={left_place_hover_candidate_xyzs} | "
            f"left_reference_pose_xyz={DrawerOpenPlaceActionBank._as_numpy(env, left_reference_pose[:3, 3]).tolist()} | "
            f"left_reference_pose_rot={DrawerOpenPlaceActionBank._as_numpy(env, left_reference_pose[:3, :3]).tolist()} | "
            f"left_lift_pose_xyz={DrawerOpenPlaceActionBank._as_numpy(env, left_lift_pose[:3, 3]).tolist()} | "
            f"left_lift_pose_rot={DrawerOpenPlaceActionBank._as_numpy(env, left_lift_pose[:3, :3]).tolist()} | "
            f"left_plan_lift_last_qpos={DrawerOpenPlaceActionBank._as_numpy(env, left_plan_lift.positions[-1]).tolist()} | "
            f"left_home_qpos={DrawerOpenPlaceActionBank._as_numpy(env, left_home_qpos).tolist()}"
        )
        left_place_hover_result = DrawerOpenPlaceActionBank._solve_ik_candidates(
            env,
            "left_arm",
            left_place_hover_candidates,
            [left_lift_qpos, left_home_qpos],
            "left_place_hover",
        )
        if left_place_hover_result is None:
            return False
        _, left_place_hover_qpos = left_place_hover_result
        left_place_hover_pose = DrawerOpenPlaceActionBank._fk_pose(
            env, "left_arm", left_place_hover_qpos
        )

        left_place_drop_pose = left_place_hover_pose.clone()
        left_place_drop_pose[2, 3] = place_drop_xyz[2]
        left_plan_to_place_drop = DrawerOpenPlaceActionBank._require_plan(
            env,
            DrawerOpenPlaceActionBank._plan_linear_eef_motion(
                env,
                "left_arm",
                left_place_hover_qpos,
                [left_place_hover_pose, left_place_drop_pose],
                sample_interval=50,
            ),
            "left_place_drop",
        )
        if left_plan_to_place_drop is None:
            return False
        left_place_drop_qpos = left_plan_to_place_drop.positions[-1]

        affordance_datas["left_arm_init_qpos"] = left_home_qpos
        affordance_datas["left_arm_duck_hover_pose"] = left_hover_pose
        affordance_datas["left_arm_duck_hover_qpos"] = left_hover_qpos
        affordance_datas["left_arm_duck_pregrasp_pose"] = left_pregrasp_pose
        affordance_datas["left_arm_duck_pregrasp_qpos"] = left_pregrasp_qpos
        affordance_datas["left_arm_duck_grasp_pose"] = left_grasp_pose
        affordance_datas["left_arm_duck_grasp_qpos"] = left_grasp_qpos
        affordance_datas["left_arm_duck_lift_pose"] = left_lift_pose
        affordance_datas["left_arm_duck_lift_qpos"] = left_lift_qpos
        affordance_datas["left_arm_place_hover_pose"] = left_place_hover_pose
        affordance_datas["left_arm_place_hover_qpos"] = left_place_hover_qpos
        affordance_datas["left_arm_place_drop_pose"] = left_place_drop_pose
        affordance_datas["left_arm_place_drop_qpos"] = left_place_drop_qpos
        affordance_datas["left_arm_retreat_pose"] = left_place_hover_pose
        affordance_datas["left_arm_retreat_qpos"] = left_place_hover_qpos
        affordance_datas["_left_arm_pick_place_plan_ready"] = True
        return True

    @staticmethod
    def _repeat_qpos(env, qpos: torch.Tensor, duration: int) -> np.ndarray:
        if duration <= 0:
            duration = 1
        qpos_np = DrawerOpenPlaceActionBank._as_numpy(env, qpos).reshape(-1)
        return np.repeat(qpos_np[:, None], duration, axis=1)

    @staticmethod
    def _match_control_dim(env, control_part: str, qpos) -> torch.Tensor:
        qpos_tensor = DrawerOpenPlaceActionBank._as_tensor(env, qpos).reshape(-1)
        target_dim = len(env.robot.get_joint_ids(name=control_part, remove_mimic=True))
        if target_dim <= 0:
            return qpos_tensor
        if qpos_tensor.numel() == target_dim:
            return qpos_tensor
        return qpos_tensor[:target_dim]

    @staticmethod
    def _get_eef_limit_qpos(env, control_part: str, is_open: bool) -> torch.Tensor:
        qpos_limits = env.robot.get_qpos_limits(name=control_part)
        limit_idx = 1 if is_open else 0
        limit_qpos = DrawerOpenPlaceActionBank._as_tensor(env, qpos_limits)[0, :, limit_idx]
        return DrawerOpenPlaceActionBank._match_control_dim(
            env, control_part, limit_qpos
        )

    @staticmethod
    @tag_node
    @resolve_env_params
    def capture_current_arm_state(env, control_part: str, **kwargs):
        affordance_datas = DrawerOpenPlaceActionBank._ensure_affordance_datas(env)
        qpos = DrawerOpenPlaceActionBank._as_tensor(env, env.robot.get_qpos(control_part)[0])
        affordance_datas[f"{control_part}_init_qpos"] = qpos
        affordance_datas[f"{control_part}_init_pose"] = DrawerOpenPlaceActionBank._fk_pose(
            env, control_part, qpos
        )
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def prepare_eef_qpos_limits(env, control_part: str, **kwargs):
        affordance_datas = DrawerOpenPlaceActionBank._ensure_affordance_datas(env)
        open_qpos = DrawerOpenPlaceActionBank._get_eef_limit_qpos(
            env, control_part, is_open=True
        )
        close_qpos = DrawerOpenPlaceActionBank._get_eef_limit_qpos(
            env, control_part, is_open=False
        )
        affordance_datas[f"{control_part}_init_qpos"] = DrawerOpenPlaceActionBank._as_numpy(
            env, open_qpos
        )
        affordance_datas[f"{control_part}_open_qpos"] = open_qpos
        affordance_datas[f"{control_part}_close_qpos"] = close_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_right_drawer_node(env, node_key: str, **kwargs):
        if DrawerOpenPlaceActionBank._ensure_right_arm_plan(env) is False:
            return False
        return node_key in env.affordance_datas

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_pick_place_node(env, node_key: str, **kwargs):
        if DrawerOpenPlaceActionBank._ensure_left_arm_plan(env) is False:
            return False
        return node_key in env.affordance_datas

    @staticmethod
    @tag_node
    @tag_edge
    @resolve_env_params
    def execute_open(
        env,
        control_part: str | None = None,
        return_action: bool = False,
        duration: int = 1,
        **kwargs,
    ):
        if return_action is False:
            return True
        if control_part is None:
            raise ValueError("execute_open requires control_part for real gripper qpos.")
        affordance_datas = DrawerOpenPlaceActionBank._ensure_affordance_datas(env)
        start_qpos = affordance_datas.get(f"{control_part}_close_qpos")
        target_qpos = affordance_datas.get(f"{control_part}_open_qpos")
        if start_qpos is None or target_qpos is None:
            start_qpos = DrawerOpenPlaceActionBank._get_eef_limit_qpos(
                env, control_part, is_open=False
            )
            target_qpos = DrawerOpenPlaceActionBank._get_eef_limit_qpos(
                env, control_part, is_open=True
            )
        start_qpos = DrawerOpenPlaceActionBank._as_tensor(env, start_qpos)
        target_qpos = DrawerOpenPlaceActionBank._as_tensor(env, target_qpos)
        interpolated = DrawerOpenPlaceActionBank._interpolate_state(
            env, start_qpos, target_qpos, duration
        )
        return DrawerOpenPlaceActionBank._as_numpy(env, interpolated.T)

    @staticmethod
    @tag_node
    @tag_edge
    @resolve_env_params
    def execute_close(
        env,
        control_part: str | None = None,
        return_action: bool = False,
        duration: int = 1,
        **kwargs,
    ):
        if return_action is False:
            return True
        if control_part is None:
            raise ValueError("execute_close requires control_part for real gripper qpos.")
        affordance_datas = DrawerOpenPlaceActionBank._ensure_affordance_datas(env)
        start_qpos = affordance_datas.get(f"{control_part}_open_qpos")
        target_qpos = affordance_datas.get(f"{control_part}_close_qpos")
        if start_qpos is None or target_qpos is None:
            start_qpos = DrawerOpenPlaceActionBank._get_eef_limit_qpos(
                env, control_part, is_open=True
            )
            target_qpos = DrawerOpenPlaceActionBank._get_eef_limit_qpos(
                env, control_part, is_open=False
            )
        start_qpos = DrawerOpenPlaceActionBank._as_tensor(env, start_qpos)
        target_qpos = DrawerOpenPlaceActionBank._as_tensor(env, target_qpos)
        interpolated = DrawerOpenPlaceActionBank._interpolate_state(
            env, start_qpos, target_qpos, duration
        )
        return DrawerOpenPlaceActionBank._as_numpy(env, interpolated.T)

    @staticmethod
    @tag_edge
    @resolve_env_params
    def execute_joint_motion(
        env,
        control_part: str,
        start_key: str,
        target_key: str,
        duration: int,
        stage_name: str = "",
        **kwargs,
    ):
        start_qpos = DrawerOpenPlaceActionBank._as_tensor(
            env, env.affordance_datas[start_key]
        )
        target_qpos = DrawerOpenPlaceActionBank._as_tensor(
            env, env.affordance_datas[target_key]
        )
        plan_result = DrawerOpenPlaceActionBank._require_plan(
            env,
            DrawerOpenPlaceActionBank._plan_joint_motion(
                env,
                control_part,
                start_qpos,
                target_qpos,
                sample_interval=duration,
            ),
            stage_name or f"{control_part}_{start_key}_to_{target_key}",
        )
        if plan_result is None:
            return DrawerOpenPlaceActionBank._repeat_qpos(env, start_qpos, duration)
        return DrawerOpenPlaceActionBank._as_numpy(env, plan_result.positions.T)

    @staticmethod
    @tag_edge
    @resolve_env_params
    def execute_linear_motion(
        env,
        control_part: str,
        start_key: str,
        pose_keys: Sequence[str],
        duration: int,
        stage_name: str = "",
        **kwargs,
    ):
        start_qpos = DrawerOpenPlaceActionBank._as_tensor(
            env, env.affordance_datas[start_key]
        )
        poses = [
            DrawerOpenPlaceActionBank._as_tensor(env, env.affordance_datas[key])
            for key in pose_keys
        ]
        plan_result = DrawerOpenPlaceActionBank._require_plan(
            env,
            DrawerOpenPlaceActionBank._plan_linear_eef_motion(
                env,
                control_part,
                start_qpos,
                poses,
                sample_interval=duration,
            ),
            stage_name or f"{control_part}_{pose_keys[-1]}",
        )
        if plan_result is None:
            return DrawerOpenPlaceActionBank._repeat_qpos(env, start_qpos, duration)
        return DrawerOpenPlaceActionBank._as_numpy(env, plan_result.positions.T)

    @staticmethod
    @tag_edge
    @resolve_env_params
    def execute_hold(
        env,
        control_part: str,
        qpos_key: str,
        duration: int,
        **kwargs,
    ):
        qpos = DrawerOpenPlaceActionBank._as_tensor(env, env.affordance_datas[qpos_key])
        return DrawerOpenPlaceActionBank._repeat_qpos(env, qpos, duration)
