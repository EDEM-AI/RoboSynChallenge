import numpy as np
import torch
from copy import deepcopy
from typing import List

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
)
from embodichain.utils import logger

__all__ = ["DrawerOpenPlaceActionBank"]


class DrawerOpenPlaceActionBank(ActionBank):
    """Small task-specific action bank for DrawerOpenPlace.

    The geometric recipe is intentionally kept in action_config.json. This class
    only mirrors the lightweight style used by tasks such as beaker_mixer:
    generate a simple arm-facing seed, command grippers, and plan joint paths.
    """

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _first_pose(value) -> np.ndarray:
        pose = DrawerOpenPlaceActionBank._to_numpy(value)
        if pose.ndim == 3:
            return pose[0]
        return pose

    @staticmethod
    def _interpolate_qpos(start_qpos, target_qpos, duration: int) -> np.ndarray:
        start_qpos = DrawerOpenPlaceActionBank._to_numpy(start_qpos).reshape(-1)
        target_qpos = DrawerOpenPlaceActionBank._to_numpy(target_qpos).reshape(-1)
        duration = max(int(duration), 1)
        if duration == 1:
            return target_qpos[:, None]
        alpha = np.linspace(0.0, 1.0, duration, dtype=np.float32)[:, None]
        qpos = start_qpos[None, :] * (1.0 - alpha) + target_qpos[None, :] * alpha
        return qpos.T.astype(np.float32)

    @staticmethod
    def _match_control_dim(env, control_part: str, qpos) -> np.ndarray:
        qpos = DrawerOpenPlaceActionBank._to_numpy(qpos).reshape(-1)
        target_dim = len(env.robot.get_joint_ids(name=control_part, remove_mimic=True))
        if target_dim <= 0 or qpos.shape[0] == target_dim:
            return qpos
        return qpos[:target_dim]

    @staticmethod
    def _get_eef_limit_qpos(env, control_part: str, is_open: bool) -> np.ndarray:
        qpos_limits = DrawerOpenPlaceActionBank._to_numpy(
            env.robot.get_qpos_limits(name=control_part)
        )
        limit_idx = 1 if is_open else 0
        return DrawerOpenPlaceActionBank._match_control_dim(
            env, control_part, qpos_limits[0, :, limit_idx]
        )

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_aim_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        left_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["duck_pose"][:2, 3]
                    - env.affordance_datas["left_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        left_arm_aim_qpos = deepcopy(env.affordance_datas["left_arm_init_qpos"])
        left_arm_aim_qpos[0] = left_aim_horizontal_angle
        env.affordance_datas["left_arm_aim_duck_qpos"] = left_arm_aim_qpos
        return True
    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_aim_drawer_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        left_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["drawer_pose"][:2, 3]
                    - env.affordance_datas["left_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        left_arm_aim_qpos = deepcopy(env.affordance_datas["left_arm_init_qpos"])
        left_arm_aim_qpos[0] = left_aim_horizontal_angle
        env.affordance_datas["left_arm_aim_drawer_qpos"] = left_arm_aim_qpos
        return True
    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_right_arm_aim_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        right_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["drawer_pose"][:2, 3]
                    - env.affordance_datas["right_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        right_arm_aim_qpos = deepcopy(env.affordance_datas["right_arm_init_qpos"])
        right_arm_aim_qpos[0] = right_aim_horizontal_angle
        env.affordance_datas["right_arm_aim_drawer_qpos"] = right_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def prepare_eef_qpos_limits(env, control_part: str, **kwargs):
        open_qpos = DrawerOpenPlaceActionBank._get_eef_limit_qpos(
            env, control_part, is_open=True
        )
        close_qpos = DrawerOpenPlaceActionBank._get_eef_limit_qpos(
            env, control_part, is_open=False
        )
        env.affordance_datas[f"{control_part}_init_qpos"] = open_qpos
        env.affordance_datas[f"{control_part}_open_qpos"] = open_qpos
        env.affordance_datas[f"{control_part}_close_qpos"] = close_qpos
        return True

    @staticmethod
    @tag_node
    @tag_edge
    @resolve_env_params
    def execute_open(
        env,
        control_part: str | None = None,
        return_action: bool = False,
        duration: int = 15,
        **kwargs,
    ):
        if return_action is False:
            return True
        if control_part is None:
            raise ValueError("execute_open requires control_part.")

        open_qpos = env.affordance_datas.get(
            f"{control_part}_open_qpos",
            DrawerOpenPlaceActionBank._get_eef_limit_qpos(
                env, control_part, is_open=True
            ),
        )
        close_qpos = env.affordance_datas.get(
            f"{control_part}_close_qpos",
            DrawerOpenPlaceActionBank._get_eef_limit_qpos(
                env, control_part, is_open=False
            ),
        )
        close_ratio = float(env.affordance_datas.get(f"{control_part}_close_ratio", 1.0))
        close_ratio = max(0.0, min(1.0, close_ratio))
        start_qpos = DrawerOpenPlaceActionBank._to_numpy(open_qpos) + (
            DrawerOpenPlaceActionBank._to_numpy(close_qpos)
            - DrawerOpenPlaceActionBank._to_numpy(open_qpos)
        ) * close_ratio
        return DrawerOpenPlaceActionBank._interpolate_qpos(
            start_qpos, open_qpos, duration
        )

    @staticmethod
    @tag_node
    @tag_edge
    @resolve_env_params
    def execute_close(
        env,
        control_part: str | None = None,
        return_action: bool = False,
        duration: int = 15,
        **kwargs,
    ):
        if return_action is False:
            return True
        if control_part is None:
            raise ValueError("execute_close requires control_part.")

        open_qpos = env.affordance_datas.get(
            f"{control_part}_open_qpos",
            DrawerOpenPlaceActionBank._get_eef_limit_qpos(
                env, control_part, is_open=True
            ),
        )
        close_qpos = env.affordance_datas.get(
            f"{control_part}_close_qpos",
            DrawerOpenPlaceActionBank._get_eef_limit_qpos(
                env, control_part, is_open=False
            ),
        )
        close_ratio = float(env.affordance_datas.get(f"{control_part}_close_ratio", 1.0))
        close_ratio = max(0.0, min(1.0, close_ratio))
        target_qpos = DrawerOpenPlaceActionBank._to_numpy(open_qpos) + (
            DrawerOpenPlaceActionBank._to_numpy(close_qpos)
            - DrawerOpenPlaceActionBank._to_numpy(open_qpos)
        ) * close_ratio
        return DrawerOpenPlaceActionBank._interpolate_qpos(
            open_qpos, target_qpos, duration
        )

    @staticmethod
    @tag_edge
    def plan_trajectory(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
        edge_name: str = "",
    ):
        keyposes = [
            DrawerOpenPlaceActionBank._to_numpy(env.affordance_datas[keypose_name])
            for keypose_name in keypose_names
        ]

        if all(
            np.linalg.norm(former - latter).sum() <= 1e-3
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            logger.log_warning(
                "Applying plan_trajectory to close qpos values. Using stand_still."
            )
            return DrawerOpenPlaceActionBank.stand_still(
                env, agent_uid, keypose_names, duration
            )

        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
        )
        plan_state = [
            PlanState(qpos=torch.as_tensor(qpos), move_type=MoveType.JOINT_MOVE)
            for qpos in keyposes
        ]

        ret = motion_generator.generate(
            target_states=plan_state,
            options=MotionGenOptions(
                control_part=agent_uid,
                plan_opts=ToppraPlanOptions(sample_interval=duration),
            ),
        )
        if ret is None or ret.positions is None:
            logger.log_warning(
                f"Motion plan failed for {edge_name or keypose_names}; using stand_still."
            )
            return DrawerOpenPlaceActionBank.stand_still(
                env, agent_uid, keypose_names, duration
            )
        positions = ret.positions
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        return positions.T.astype(np.float32)

    @staticmethod
    @tag_edge
    def stand_still(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
    ):
        qpos = DrawerOpenPlaceActionBank._to_numpy(
            env.affordance_datas[keypose_names[0]]
        ).reshape(-1)
        target_dim = len(env.robot.get_joint_ids(agent_uid, remove_mimic=True))
        if qpos.shape[0] != target_dim:
            logger.log_error(
                f"The shape of stand_still qpos is different from {agent_uid}'s setting."
            )
        return np.asarray([qpos] * max(int(duration), 1), dtype=np.float32).T
