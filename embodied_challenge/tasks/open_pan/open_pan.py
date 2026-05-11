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
from embodied_challenge.managers.events import visualize_rigid_body_pose
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

from .action_bank import OpenPanActionBank

__all__ = [
    "OpenPanEnv",
    "OpenPanAgentEnv",
    "OpenPanPickAndPlaceEnv",
    "OpenPanPickAndPlaceAgentEnv",
]


@register_env("OpenPanPickAndPlaceEnv-v1", max_episode_steps=600)
class OpenPanEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        self.action_config = kwargs.get("action_config", None)
        self.affordance_datas: Dict[str, np.ndarray] = {}

        self.lid_grasp_pose_object = np.eye(4, dtype=np.float32)
        self.carrot_grasp_pose_object = np.eye(4, dtype=np.float32)
        self.apple_grasp_pose_object = np.eye(4, dtype=np.float32)
        self.lid_grasp_offset = 0.0
        self.carrot_grasp_offset = 0.0
        self.apple_grasp_offset = 0.0
        self.lid_pose_orig = np.eye(4, dtype=np.float32)
        self.carrot_pose_orig = np.eye(4, dtype=np.float32)
        self.apple_pose_orig = np.eye(4, dtype=np.float32)

        self.lid_xy_random_center = np.zeros(2, dtype=np.float32)
        self.carrot_xy_random_center = np.zeros(2, dtype=np.float32)
        self.apple_xy_random_center = np.zeros(2, dtype=np.float32)

        self.agent_qpos_flip_ids = [3, 4]
        self.agent_qpos_flip_threshold = 3.455751918948773
        self.agent_qpos_flip_mode = "delta"

    @staticmethod
    def _to_matrix4(data: np.ndarray | torch.Tensor | list | tuple) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim == 3:
            arr = arr[0]
        return arr.astype(np.float32)

    def _get_rigid_pose(self, uid: str) -> np.ndarray | None:
        try:
            obj = self.sim.get_rigid_object(uid)
            pose = obj.get_local_pose(to_matrix=True)
            if isinstance(pose, torch.Tensor):
                pose = pose.squeeze(0).cpu().numpy()
            else:
                pose = np.asarray(pose).squeeze(0)
            return pose.astype(np.float32)
        except Exception:
            return None

    def _sync_open_pan_runtime_attrs(self) -> None:
        aff = self.affordance_datas

        lid_pose = self._get_rigid_pose("lid")
        carrot_pose = self._get_rigid_pose("carrot")
        if carrot_pose is None:
            carrot_pose = self._get_rigid_pose("apple")

        if lid_pose is not None:
            self.lid_pose_orig = lid_pose
            aff["lid_pose_orig"] = lid_pose
            aff["lid_pose"] = lid_pose

        if carrot_pose is not None:
            self.carrot_pose_orig = carrot_pose
            self.apple_pose_orig = carrot_pose
            aff["carrot_pose_orig"] = carrot_pose
            aff["carrot_pose"] = carrot_pose
            aff["apple_pose_orig"] = carrot_pose
            aff["apple_pose"] = carrot_pose

        self.lid_xy_random_center = np.asarray(self.lid_pose_orig[:2, 3], dtype=np.float32)
        self.carrot_xy_random_center = np.asarray(
            self.carrot_pose_orig[:2, 3], dtype=np.float32
        )
        self.apple_xy_random_center = np.asarray(self.apple_pose_orig[:2, 3], dtype=np.float32)

        if "lid_grasp_pose_object" in aff:
            self.lid_grasp_pose_object = self._to_matrix4(aff["lid_grasp_pose_object"])
        if "carrot_grasp_pose_object" in aff:
            self.carrot_grasp_pose_object = self._to_matrix4(aff["carrot_grasp_pose_object"])
            self.apple_grasp_pose_object = self.carrot_grasp_pose_object
        elif "apple_grasp_pose_object" in aff:
            self.carrot_grasp_pose_object = self._to_matrix4(aff["apple_grasp_pose_object"])
            self.apple_grasp_pose_object = self.carrot_grasp_pose_object

        self.lid_grasp_offset = float(
            aff.get("lid_grasp_offset", getattr(self, "lid_grasp_offset", 0.0))
        )
        self.carrot_grasp_offset = float(
            aff.get("carrot_grasp_offset", aff.get("apple_grasp_offset", getattr(self, "carrot_grasp_offset", 0.0)))
        )
        self.apple_grasp_offset = self.carrot_grasp_offset

        aff["lid_grasp_pose_object"] = self.lid_grasp_pose_object
        aff["carrot_grasp_pose_object"] = self.carrot_grasp_pose_object
        aff["apple_grasp_pose_object"] = self.apple_grasp_pose_object
        aff["carrot_grasp_offset"] = self.carrot_grasp_offset
        aff["apple_grasp_offset"] = self.apple_grasp_offset

    def get_arm_fk(
        self, qpos: np.ndarray, control_part: str, is_world_coordinates: bool = True
    ) -> np.ndarray:
        xpos = self.robot.compute_fk(
            name=control_part, qpos=torch.as_tensor(qpos), to_matrix=True
        ).squeeze(0)
        return xpos.cpu().numpy()

    def get_arm_ik(
        self,
        target_xpos: np.ndarray,
        is_left: bool,
        qpos_seed: np.ndarray = None,
    ) -> Tuple[bool, np.ndarray]:
        xpos = torch.as_tensor(target_xpos, dtype=torch.float32, device=self.device)
        control_part = "left_arm" if is_left else "right_arm"
        seed = (
            None
            if qpos_seed is None
            else torch.as_tensor(qpos_seed, dtype=torch.float32, device=self.device)
        )

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

    def create_demo_action_list(self, *args, **kwargs):
        logger.log_info("Create demo action list for OpenPanTask.")

        if self.action_config is None:
            logger.log_error("No action_config found in env, please check again.")

        self._sync_open_pan_runtime_attrs()
        self._init_action_bank(OpenPanActionBank, self.action_config)
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
        try:
            pan = self.sim.get_rigid_object("pan")
            lid = self.sim.get_rigid_object("lid")
            carrot = self.sim.get_rigid_object("carrot")
            if carrot is None:
                carrot = self.sim.get_rigid_object("apple")

            pan_pose = pan.get_local_pose(to_matrix=True)
            lid_pose = lid.get_local_pose(to_matrix=True)
            carrot_pose = carrot.get_local_pose(to_matrix=True)

            pan_xy = pan_pose[:, :2, 3]
            lid_xy = lid_pose[:, :2, 3]
            carrot_xy = carrot_pose[:, :2, 3]

            carrot_pan_dist = torch.linalg.norm(carrot_xy - pan_xy, dim=-1)
            lid_pan_dist = torch.linalg.norm(lid_xy - pan_xy, dim=-1)

            pan_z = pan_pose[:, 2, 3]
            lid_z = lid_pose[:, 2, 3]
            carrot_z = carrot_pose[:, 2, 3]

            carrot_in_pan_region = (carrot_pan_dist < 0.10) & (carrot_z > pan_z - 0.03)
            lid_back_on_pan = (lid_pan_dist < 0.08) & (lid_z > pan_z)

            return carrot_in_pan_region & lid_back_on_pan
        except Exception:
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)


@register_env("OpenPanPickAndPlaceAgent-v1", max_episode_steps=600)
class OpenPanAgentEnv(BaseAgentEnv, OpenPanEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info


# Backward-compatible aliases matching legacy naming.
OpenPanPickAndPlaceEnv = OpenPanEnv
OpenPanPickAndPlaceAgentEnv = OpenPanAgentEnv
