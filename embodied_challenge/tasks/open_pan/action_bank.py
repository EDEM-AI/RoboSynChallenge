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

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch

from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
    tag_edge,
    tag_node,
)
from embodichain.lab.gym.utils.misc import (
    get_changed_pose,
    mul_linear_expand,
    resolve_env_params,
    validation_with_process_from_name,
)
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    PlanState,
    ToppraPlanOptions,
    ToppraPlannerCfg,
)
from embodichain.utils import logger

__all__ = ["OpenPanActionBank"]


class OpenPanActionBank(ActionBank):
    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_aim_qpos(
        env,
        valid_funcs_name_kwargs_proc: List | None = None,
    ):
        left_aim_horizontal_angle = np.arctan2(
            *((env.affordance_datas["lid_pose"][:2, 3] - env.affordance_datas["left_arm_base_pose"][:2, 3])[1::-1])
        )
        left_arm_aim_qpos = deepcopy(env.affordance_datas["left_arm_init_qpos"])
        left_arm_aim_qpos[0] = left_aim_horizontal_angle
        env.affordance_datas["left_arm_aim_qpos"] = left_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_right_arm_aim_qpos(
        env,
        valid_funcs_name_kwargs_proc: List | None = None,
    ):
        right_aim_horizontal_angle = np.arctan2(
            *((env.affordance_datas["apple_pose"][:2, 3] - env.affordance_datas["right_arm_base_pose"][:2, 3])[1::-1])
        )
        right_arm_aim_qpos = deepcopy(env.affordance_datas["right_arm_init_qpos"])
        right_arm_aim_qpos[0] = right_aim_horizontal_angle
        env.affordance_datas["right_arm_aim_qpos"] = right_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def compute_unoffset_for_exp(env, pose_input_output_names_changes: Dict = {}):
        # Unified unoffset entry compatible with new config naming.
        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )
        return True

    @staticmethod
    @tag_node
    def generate_left_arm_init_qpos(env):
        left_arm_init_pose = env._get_arm_fk(
            env.affordance_datas["left_arm_init_qpos"],
            uid="left_arm",
            is_world_coordinates=True,
        )
        env.affordance_datas["left_arm_init_pose"] = np.asarray(left_arm_init_pose)
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_grasp(env, valid_funcs_name_kwargs_proc: List | None = None):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        lid_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.lid_grasp_pose_object),
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_grasp_pose is None:
            return False
        env.affordance_datas["lid_grasp_pose"] = lid_grasp_pose

        lid_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_grasp_qpos is None:
            return False

        env.affordance_datas["lid_grasp_qpos"] = lid_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_pre2_grasp_qpos(
        env, valid_funcs_name_kwargs_proc: List | None = None
    ):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        lid_pre2_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_pre2_grasp_pose is None:
            return False
        env.affordance_datas["lid_pre2_grasp_pose"] = lid_pre2_grasp_pose

        lid_pre2_grasp_qpos = validation_with_process_from_name(
            env,
            env.affordance_datas["lid_pre2_grasp_pose"],
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_pre2_grasp_qpos is None:
            logger.log_warning("Failed to generate lid_pre2_grasp_qpos")
            return False
        env.affordance_datas["lid_pre2_grasp_qpos"] = lid_pre2_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_pre1_grasp_qpos(
        env, valid_funcs_name_kwargs_proc: List | None = None
    ):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        lid_pre1_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_pre1_grasp_pose is None:
            return False
        env.affordance_datas["lid_pre1_grasp_pose"] = lid_pre1_grasp_pose

        lid_pre1_grasp_qpos = validation_with_process_from_name(
            env,
            lid_pre1_grasp_pose,
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_pre1_grasp_qpos is None:
            logger.log_warning("Failed to generate lid_pre1_grasp_qpos")
            return False

        env.affordance_datas["lid_pre1_grasp_qpos"] = lid_pre1_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_lift_qpos(env, valid_funcs_name_kwargs_proc: List | None = None):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        lid_lift_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_lift_pose is None:
            return False
        env.affordance_datas["lid_lift_pose"] = lid_lift_pose

        lid_lift_qpos = validation_with_process_from_name(
            env,
            env.affordance_datas["lid_lift_pose"],
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_lift_qpos is None:
            logger.log_warning("Failed to generate lid_lift_qpos")
            return False
        env.affordance_datas["lid_lift_qpos"] = lid_lift_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_aside_qpos(env, valid_funcs_name_kwargs_proc: List | None = None):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        lid_aside_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_lift_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_aside_pose is None:
            return False
        env.affordance_datas["lid_aside_pose"] = lid_aside_pose

        lid_aside_qpos = validation_with_process_from_name(
            env,
            lid_aside_pose,
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_aside_qpos is None:
            logger.log_warning("Failed to generate lid_aside_qpos")
            return False
        env.affordance_datas["lid_aside_qpos"] = lid_aside_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_back_qpos(env, valid_funcs_name_kwargs_proc: List | None = None):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        lid_back_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_back_pose is None:
            return False
        env.affordance_datas["lid_back_pose"] = lid_back_pose

        lid_back_qpos = validation_with_process_from_name(
            env,
            lid_back_pose,
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_back_qpos is None:
            logger.log_warning("Failed to generate lid_back_qpos")
            return False
        env.affordance_datas["lid_back_qpos"] = lid_back_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def left_arm_compute_unoffset_for_exp(env, pose_input_output_names_changes: Dict = {}):
        env.affordance_datas["lid_grasp_unoffset_matrix_object"] = np.eye(4)
        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )
        return True

    @staticmethod
    @tag_node
    def generate_right_arm_init_qpos(env):
        right_arm_init_pose = env._get_arm_fk(
            env.affordance_datas["right_arm_init_qpos"],
            uid="right_arm",
            is_world_coordinates=True,
        )
        env.affordance_datas["right_arm_init_pose"] = np.asarray(right_arm_init_pose)
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_grasp_qpos(env, valid_funcs_name_kwargs_proc: List | None = None):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        apple_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.apple_grasp_pose_object),
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_grasp_pose is None:
            return False
        env.affordance_datas["apple_grasp_pose"] = apple_grasp_pose

        apple_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_grasp_qpos is None:
            return False
        env.affordance_datas["apple_grasp_qpos"] = apple_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_pre1_grasp_pose_qpos(
        env, valid_funcs_name_kwargs_proc: List | None = None
    ):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        apple_pre1_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_pre1_grasp_pose is None:
            return False
        env.affordance_datas["apple_pre1_grasp_pose"] = apple_pre1_grasp_pose

        apple_pre1_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_pre1_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_pre1_grasp_qpos is None:
            return False
        env.affordance_datas["apple_pre1_grasp_qpos"] = apple_pre1_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_pre2_grasp_qpos(
        env, valid_funcs_name_kwargs_proc: List | None = None
    ):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        apple_pre2_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_pre2_grasp_pose is None:
            return False
        env.affordance_datas["apple_pre2_grasp_pose"] = apple_pre2_grasp_pose

        apple_pre2_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_pre2_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_pre2_grasp_qpos is None:
            logger.log_warning("Failed to generate apple_pre2_grasp_qpos")
            return False
        env.affordance_datas["apple_pre2_grasp_qpos"] = apple_pre2_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_lift_qpos(env, valid_funcs_name_kwargs_proc: List | None = None):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        apple_lift_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_lift_pose is None:
            return False
        env.affordance_datas["apple_lift_pose"] = apple_lift_pose

        apple_lift_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_lift_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_lift_qpos is None:
            logger.log_warning("Failed to generate apple_lift_qpos")
            return False
        env.affordance_datas["apple_lift_qpos"] = apple_lift_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_to_pan_qpos(
        env, valid_funcs_name_kwargs_proc: List | None = None
    ):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        apple_to_pan_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_lift_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_to_pan_pose is None:
            return False
        env.affordance_datas["apple_to_pan_pose"] = apple_to_pan_pose

        apple_to_pan_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_to_pan_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_to_pan_qpos is None:
            logger.log_warning("Failed to generate apple_to_pan_qpos")
            return False
        env.affordance_datas["apple_to_pan_qpos"] = apple_to_pan_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_after_pan_qpos(
        env, valid_funcs_name_kwargs_proc: List | None = None
    ):
        if valid_funcs_name_kwargs_proc is None:
            valid_funcs_name_kwargs_proc = []

        apple_after_pan_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_to_pan_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_after_pan_pose is None:
            return False
        env.affordance_datas["apple_after_pan_pose"] = apple_after_pan_pose

        apple_after_pan_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_after_pan_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_after_pan_qpos is None:
            logger.log_warning("Failed to generate apple_after_pan_qpos")
            return False
        env.affordance_datas["apple_after_pan_qpos"] = apple_after_pan_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def right_arm_compute_unoffset_for_exp(
        env, pose_input_output_names_changes: Dict = {}
    ):
        env.affordance_datas["apple_grasp_unoffset_matrix_object"] = np.eye(4)
        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )
        return True

    @staticmethod
    @tag_edge
    @tag_node
    def execute_open(env, return_action: bool = False, **kwargs):
        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                action = mul_linear_expand(np.array([[0.0], [1.0]]), [duration - 1])
                action = np.concatenate([action, np.array([[1.0]])]).transpose()
            else:
                action = np.ones((1, duration))
            return action
        return True

    @staticmethod
    @tag_edge
    @tag_node
    def execute_close(env, return_action: bool = False, **kwargs):
        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                action = mul_linear_expand(np.array([[1.0], [0.0]]), [duration - 1])
                action = np.concatenate([action, np.array([[0.0]])]).transpose()
            else:
                action = np.zeros((1, duration))
            return action
        return True

    @staticmethod
    @tag_edge
    def stand_still(env, agent_uid: str, keypose_names: List[str], duration: int):
        keyposes = [env.affordance_datas[keypose_name] for keypose_name in keypose_names]
        stand_still_qpos = keyposes[0]

        if stand_still_qpos.shape != np.asarray(env.robot.get_joint_ids("left_arm")).shape:
            logger.log_error(
                f"The shape of stand_still qpos is different from {agent_uid}'s setting."
            )

        if any(
            np.linalg.norm(former - latter).sum() > 1e-6
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            logger.log_warning(
                f"Applying stand still to two different qpos! Using the first qpos {stand_still_qpos}"
            )

        ret = np.asarray([stand_still_qpos] * duration)
        return ret.T

    @staticmethod
    @tag_edge
    def plan_trajectory(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
        edge_name: str = "",
    ):
        keyposes = [env.affordance_datas[keypose_name] for keypose_name in keypose_names]
        keyposes = [
            kp.cpu().numpy() if hasattr(kp, "cpu") and hasattr(kp, "numpy") else kp
            for kp in keyposes
        ]

        if all(
            np.linalg.norm(former - latter).sum() <= 1e-3
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            logger.log_warning(
                "Applying plan_trajectory to two very close qpos! Using stand_still."
            )
            return OpenPanActionBank.stand_still(
                env,
                agent_uid,
                keypose_names,
                duration,
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
                plan_opts=ToppraPlanOptions(
                    sample_interval=duration,
                ),
            ),
        )

        return ret.positions.numpy().T
