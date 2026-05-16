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
import numpy as np
from copy import deepcopy
from typing import List

from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
    tag_node,
    tag_edge,
)

from embodichain.lab.gym.utils.misc import resolve_env_params, mul_linear_expand

from embodichain.lab.sim.planners import (
    MoveType,
    PlanState,
    MotionGenerator,
    MotionGenCfg,
    MotionGenOptions,
    ToppraPlanOptions,
    ToppraPlannerCfg,
)
from embodichain.utils import logger


__all__ = ["MixerOperatingActionBank"]


class MixerOperatingActionBank(ActionBank):
    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_right_arm_aim_beaker_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        logger.log_warning(
            "CAUTION=============================THIS FUNC generate_right_arm_aim_beaker_qpos IS GEOMETRY-HEURISTIC."
        )
        right_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["beaker_pose"][:2, 3]
                    - env.affordance_datas["right_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        right_arm_aim_qpos = deepcopy(env.affordance_datas["right_arm_init_qpos"])
        right_arm_aim_qpos[0] = right_aim_horizontal_angle
        env.affordance_datas["right_arm_aim_beaker_qpos"] = right_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_right_arm_aim_mixer_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        logger.log_warning(
            "CAUTION=============================THIS FUNC generate_right_arm_aim_mixer_qpos IS GEOMETRY-HEURISTIC."
        )
        right_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["beaker_mixer_pose"][:2, 3]
                    - env.affordance_datas["right_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        right_arm_aim_qpos = deepcopy(env.affordance_datas["right_arm_init_qpos"])
        right_arm_aim_qpos[0] = right_aim_horizontal_angle
        env.affordance_datas["right_arm_aim_mixer_qpos"] = right_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_aim_beaker_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        logger.log_warning(
            "CAUTION=============================THIS FUNC generate_left_arm_aim_beaker_qpos IS GEOMETRY-HEURISTIC."
        )
        left_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["beaker_pose"][:2, 3]
                    - env.affordance_datas["left_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        left_arm_aim_qpos = deepcopy(env.affordance_datas["left_arm_init_qpos"])
        left_arm_aim_qpos[0] = left_aim_horizontal_angle
        env.affordance_datas["left_arm_aim_beaker_qpos"] = left_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_aim_mixer_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        logger.log_warning(
            "CAUTION=============================THIS FUNC generate_left_arm_aim_mixer_qpos IS GEOMETRY-HEURISTIC."
        )
        left_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["beaker_mixer_pose"][:2, 3]
                    - env.affordance_datas["left_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        left_arm_aim_qpos = deepcopy(env.affordance_datas["left_arm_init_qpos"])
        left_arm_aim_qpos[0] = left_aim_horizontal_angle
        env.affordance_datas["left_arm_aim_mixer_qpos"] = left_arm_aim_qpos
        return True

    @staticmethod
    @tag_edge
    @tag_node
    def execute_open(env, return_action: bool = False, **kwargs):
        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                hold_steps = 2

                if duration > hold_steps:
                    interp_steps = (duration - hold_steps) - 1
                    if interp_steps > 0:
                        interp_action = mul_linear_expand(
                            np.array([[0.0], [1.0]]), [interp_steps]
                        )
                    else:
                        interp_action = np.array([[1.0]])

                    hold_action = np.ones((hold_steps + 1, 1))

                    if interp_steps > 0:
                        action = np.concatenate(
                            [interp_action, hold_action], axis=0
                        ).transpose()
                    else:
                        action = np.concatenate(
                            [np.array([[0.0]]), np.ones((duration - 1, 1))], axis=0
                        ).transpose()
                else:
                    action = mul_linear_expand(np.array([[0.0], [1.0]]), [duration - 1])
                    action = np.concatenate([action, np.array([[1.0]])], axis=0).transpose()
            else:
                action = np.ones((1, duration))
            return action
        else:
            return True

    @staticmethod
    @tag_edge
    @tag_node
    def execute_close(env, return_action: bool = False, **kwargs):
        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                hold_steps = 5

                if duration > hold_steps:
                    interp_steps = (duration - hold_steps) - 1
                    if interp_steps > 0:
                        interp_action = mul_linear_expand(
                            np.array([[1.0], [0.0]]), [interp_steps]
                        )
                    else:
                        interp_action = np.array([[0.0]])

                    hold_action = np.zeros((hold_steps + 1, 1))

                    if interp_steps > 0:
                        action = np.concatenate(
                            [interp_action, hold_action], axis=0
                        ).transpose()
                    else:
                        action = np.concatenate(
                            [np.array([[1.0]]), np.zeros((duration - 1, 1))], axis=0
                        ).transpose()
                else:
                    action = mul_linear_expand(np.array([[1.0], [0.0]]), [duration - 1])
                    action = np.concatenate([action, np.array([[0.0]])], axis=0).transpose()
            else:
                action = np.zeros((1, duration))
            return action
        else:
            return True

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
            env.affordance_datas[keypose_name] for keypose_name in keypose_names
        ]

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
            keyposes = [keyposes[0]] * 2
            ret_transposed = MixerOperatingActionBank.stand_still(
                env,
                agent_uid,
                keypose_names,
                duration,
            )

            return ret_transposed

        else:
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

    @staticmethod
    @tag_edge
    def stand_still(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
    ):
        keyposes = [
            env.affordance_datas[keypose_name] for keypose_name in keypose_names
        ]

        stand_still_qpos = keyposes[0]

        target_joint_ids = np.asarray(env.robot.get_joint_ids(agent_uid))
        if stand_still_qpos.shape != target_joint_ids.shape:
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