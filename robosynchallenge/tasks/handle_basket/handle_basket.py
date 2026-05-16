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

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger
from robosynchallenge.managers.events import visualize_rigid_body_pose
from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from .action_bank import HandleBasketActionBank

__all__ = ["HandleBasketEnv", "HandleBasketTestEnv", "HandleBasketAgentEnv"]


@register_env("HandleBasket", max_episode_steps=600)
class HandleBasketEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        self.action_config = kwargs.get("action_config", None)
        self.affordance_datas = {}

        # Defaults used by handle_basket action config.
        self.milk_grasp_pose_object = np.eye(4, dtype=np.float32)
        self.basket_grasp_pose_object = np.eye(4, dtype=np.float32)
        self.milk_grasp_offset = 0.0
        self.basket_grasp_offset = 0.0
        self.milk_pose_orig = np.eye(4, dtype=np.float32)
        self.basket_pose_orig = np.eye(4, dtype=np.float32)
        self.milk_xy_random_center = np.zeros(2, dtype=np.float32)
        self.basket_xy_random_center = np.zeros(2, dtype=np.float32)

        self.agent_qpos_flip_ids = [3, 4]
        self.agent_qpos_flip_threshold = 3.455751918948773
        self.agent_qpos_flip_mode = "delta"
        # Keep old pose prior optional so v2 configs can use raw registered poses.
        self.use_legacy_pose_prior = bool(kwargs.get("use_legacy_pose_prior", False))

    @staticmethod
    def _to_matrix4(data: np.ndarray | torch.Tensor | list | tuple) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    @staticmethod
    def _apply_legacy_pose_prior(pose: np.ndarray, obj_name: str) -> np.ndarray:
        """Apply legacy carry_basket object-pose priors used in v1 pipeline.

        In old carry_basket scene setup:
        - basket pose was post-multiplied by Rx(90deg)
        - milk orientation was pre-rotated by Rz(-19deg)
        """
        out_pose = np.asarray(pose, dtype=np.float32).copy()

        if obj_name == "basket":
            rot_x = np.eye(4, dtype=np.float32)
            c, s = np.cos(np.deg2rad(90.0)), np.sin(np.deg2rad(90.0))
            rot_x[:3, :3] = np.array(
                [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32
            )
            out_pose = out_pose @ rot_x
        elif obj_name == "milk":
            c, s = np.cos(np.deg2rad(-19.0)), np.sin(np.deg2rad(-19.0))
            rot_z = np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
            out_pose[:3, :3] = rot_z @ out_pose[:3, :3]

        return out_pose

    def _sync_carry_basket_runtime_attrs(self) -> None:
        """Sync old-style runtime attrs from current affordances/scene state.

        The carry_basket action config still references attrs like
        `milk_xy_random_center` and `basket_xy_random_center`. In v2, these may
        stay at zeros unless we refresh them from current object poses.
        """
        aff = getattr(self, "affordance_datas", {})

        def _first_existing(*keys: str) -> np.ndarray | None:
            for key in keys:
                if key in aff:
                    return self._to_matrix4(aff[key]).astype(np.float32)
            return None

        milk_pose = _first_existing("milk_pose", "milk_pose_orig")
        basket_pose = _first_existing("basket_pose", "basket_pose_orig")
        milk_grasp_pose_obj = _first_existing(
            "milk_grasp_pose_object",
            "milk_milk_grasp_pose_object",
        )
        basket_grasp_pose_obj = _first_existing(
            "basket_grasp_pose_object",
            "basket_basket_grasp_pose_object",
        )

        if milk_pose is not None:
            self.milk_pose_orig = milk_pose
        if basket_pose is not None:
            self.basket_pose_orig = basket_pose
        if milk_grasp_pose_obj is not None:
            self.milk_grasp_pose_object = milk_grasp_pose_obj
        if basket_grasp_pose_obj is not None:
            self.basket_grasp_pose_object = basket_grasp_pose_obj

        if self.use_legacy_pose_prior:
            # Keep action config aligned with legacy carry_basket pose conventions.
            self.milk_pose_orig = self._apply_legacy_pose_prior(self.milk_pose_orig, "milk")
            self.basket_pose_orig = self._apply_legacy_pose_prior(self.basket_pose_orig, "basket")

        self.milk_xy_random_center = np.asarray(self.milk_pose_orig[:2, 3], dtype=np.float32)
        self.basket_xy_random_center = np.asarray(self.basket_pose_orig[:2, 3], dtype=np.float32)

    def get_arm_fk(
        self, qpos: np.ndarray, control_part: str, is_world_coordinates=True
    ) -> np.ndarray:
        xpos = self.robot.compute_fk(
            name=control_part, qpos=torch.as_tensor(qpos), to_matrix=True
        ).squeeze(0)

        # the xpos computed from robot is in the local arena frame, which is equivalent to world frame of the
        # old version.
        return xpos.cpu().numpy()

    def get_arm_ik(
        self,
        target_xpos: np.ndarray,
        is_left: bool,
        qpos_seed: np.ndarray = None,
    ) -> Tuple[bool, np.ndarray]:
        xpos = torch.as_tensor(target_xpos, dtype=torch.float32, device=self.device)

        control_part = "left_arm" if is_left else "right_arm"
        seed = None if qpos_seed is None else torch.as_tensor(qpos_seed, dtype=torch.float32, device=self.device)

        try:
            ret, qpos = self.robot.compute_ik(name=control_part, pose=xpos, qpos_seed=seed)
        except TypeError:
            try:
                ret, qpos = self.robot.compute_ik(name=control_part, pose=xpos, joint_seed=seed)
            except TypeError:
                ret, qpos = self.robot.compute_ik(xpos, seed, control_part)

        return ret.all().item(), qpos.squeeze(0).cpu().numpy()

    def _get_arm_fk(self, qpos: np.ndarray, uid: str, is_world_coordinates: bool = True) -> np.ndarray:
        return self.get_arm_fk(qpos=qpos, control_part=uid, is_world_coordinates=is_world_coordinates)

    def _get_arm_ik(
        self,
        target_xpos: np.ndarray,
        is_left: bool = True,
        qpos_seed: np.ndarray | None = None,
    ) -> Tuple[bool, np.ndarray]:
        return self.get_arm_ik(target_xpos=target_xpos, is_left=is_left, qpos_seed=qpos_seed)

    def action_bank_compute_ik(
        self,
        target_xpos: np.ndarray | torch.Tensor,
        qpos_seed: np.ndarray | torch.Tensor | None,
        control_part: str,
    ):
        """IK adapter for action-bank utils.get_ik_ret/get_ik_qpos.

        Expected signature is (target_xpos, qpos_seed, control_part), matching
        cached_ik() call style in gym.utils.misc.
        """
        pose = torch.as_tensor(target_xpos, dtype=torch.float32, device=self.device)
        seed = (
            None
            if qpos_seed is None
            else torch.as_tensor(qpos_seed, dtype=torch.float32, device=self.device)
        )
        return self.robot.compute_ik(pose=pose, joint_seed=seed, name=control_part)

    def adapt_cobotmagic_grasp_pose(self, pose: np.ndarray) -> np.ndarray:
        """Apply legacy CobotMagic grasp orientation adaptation.

        Old carry_basket logic remapped local grasp axes for CobotMagic so IK
        targets match gripper convention. Keep this as a no-op for other robots.
        """
        robot_uid = str(getattr(self.robot, "uid", ""))
        robot_name = self.robot.__class__.__name__
        if "cobotmagic" not in robot_uid.lower() and "cobotmagic" not in robot_name.lower():
            return pose

        adapted_pose = np.asarray(pose).copy()
        old_x = adapted_pose[:3, 0].copy()
        adapted_pose[:3, 0] = -adapted_pose[:3, 1]
        adapted_pose[:3, 1] = old_x
        return adapted_pose

    def find_nearest_valid_pose(self, pose: np.ndarray, select_arm: str, xpos_resolution: float = 0.02) -> np.ndarray:
        # Fallback implementation for configs that request this helper in rejected_processes.
        return pose

    def create_demo_action_list(self, *args, **kwargs):
        logger.log_info("Create demo action list for HandleBasket Task.")

        if self.action_config is None:
            logger.log_error("No action_config found in env, please check again.")

        self._sync_carry_basket_runtime_attrs()

        self._init_action_bank(HandleBasketActionBank, self.action_config)
        action_list = self.create_expert_demo_action_list(*args, **kwargs)

        if action_list is None:
            return action_list

        logger.log_info(
            f"Demo action list created with {len(action_list)} steps.", color="green"
        )
        return action_list

    def create_expert_demo_action_list(self, **kwargs):
        if hasattr(self, "action_bank") is False or self.action_bank is None:
            logger.log_error("Action bank is not initialized. Cannot create expert demo action list.")

        ret = self.action_bank.create_action_list(self, self.graph_compose, self.packages)

        if ret is None:
            logger.log_warning("Failed to generate expert demo action list.")
            return None

        left_arm_joints = self.robot.get_joint_ids(name="left_arm", remove_mimic=True)
        right_arm_joints = self.robot.get_joint_ids(name="right_arm", remove_mimic=True)
        left_eef_joints = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
        right_eef_joints = self.robot.get_joint_ids(name="right_eef", remove_mimic=True)

        total_traj_num = ret[list(ret.keys())[0]].shape[-1]
        num_active_joints = len(self.active_joint_ids)
        actions = torch.zeros((total_traj_num, self.num_envs, num_active_joints), dtype=torch.float32)

        global_to_active_idx = {
            joint_id: active_idx for active_idx, joint_id in enumerate(self.active_joint_ids)
        }

        for key, joints in [
            ("left_arm", left_arm_joints),
            ("left_eef", left_eef_joints),
            ("right_arm", right_arm_joints),
            ("right_eef", right_eef_joints),
        ]:
            if key in ret:
                local_action_data = torch.as_tensor(ret[key].T, dtype=torch.float32)
                for i, joint_id in enumerate(joints):
                    if joint_id in global_to_active_idx:
                        active_idx = global_to_active_idx[joint_id]
                        actions[:, 0, active_idx] = local_action_data[:, i]

        return actions
    def is_task_success(self, **kwargs) -> torch.Tensor:
        basket = self.sim.get_rigid_object("basket")
        milk = self.sim.get_rigid_object("milk")
        basket_pose = basket.get_local_pose(to_matrix=True)
        milk_pose = milk.get_local_pose(to_matrix=True)
        basket_xy = basket_pose[:, :2, 3]
        milk_xy = milk_pose[:, :2, 3]
        dist = torch.linalg.norm(milk_xy - basket_xy, dim=-1)
        up = milk_pose[:, 2, 3] > basket_pose[:, 2, 3]
        return dist < 0.12 and up

@register_env("HandleBasketTest", max_episode_steps=600)
class HandleBasketTestEnv(HandleBasketEnv):
    def compute_task_state(self, **kwargs):
    # It is difficult to determine whether a task has failed or succeeded based on conditions,
    # and manual assessment is required.
        return torch.zeros(self.num_envs, dtype=torch.bool), torch.zeros(self.num_envs, dtype=torch.bool), None
    def is_task_success(self, **kwargs):
        return torch.ones(self.num_envs, dtype=torch.bool)

@register_env("HandleBasketAgent", max_episode_steps=600)
class HandleBasketAgentEnv(BaseAgentEnv, HandleBasketEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info
