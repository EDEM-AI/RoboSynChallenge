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

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger
from robosynchallenge.managers.events import visualize_rigid_body_pose
from embodichain_tasks.tableware.base_agent_env import BaseAgentEnv
from .action_bank import HandleBasketActionBank

__all__ = [
    "HandleBasketEnv",
    "HandleBasketTestEnv",
    "HandleBasketAgentEnv",
]


@register_env("HandleBasket", max_episode_steps=600)
class HandleBasketEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)



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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._hb_diag_step = 0
        self._hb_orig_basket_x = None
        self._hb_orig_basket_y = None
        self._hb_orig_basket_z = None
        self._hb_stable_steps = 0
        self._hb_last_success_check_env_step = None
        self._hb_debug_flags = None
        self._hb_debug_last_signature = None
        self._hb_debug_check_count = 0
        self._hb_diag_path = None

        return obs, info

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Success when the carried basket moves left with milk inside stably."""
        if not hasattr(self, "_hb_stable_steps"):
            self._hb_orig_basket_x = None
            self._hb_orig_basket_y = None
            self._hb_orig_basket_z = None
            self._hb_stable_steps = 0
            self._hb_last_success_check_env_step = None
        if not hasattr(self, "_hb_diag_step"):
            self._hb_diag_step = 0

        import numpy as _np
        import torch as _torch

        basket = self.sim.get_rigid_object("basket")
        milk = self.sim.get_rigid_object("milk")
        basket_pose = basket.get_local_pose(to_matrix=True)
        milk_pose = milk.get_local_pose(to_matrix=True)

        def extract_xy_z(pose):
            arr = _np.asarray(pose)
            if arr.ndim == 3:
                return arr[:, :2, 3], arr[:, 2, 3]
            return arr[:2, 3], float(arr[2, 3])

        try:
            basket_xy, basket_z = extract_xy_z(basket_pose)
            milk_xy, milk_z = extract_xy_z(milk_pose)
        except Exception:
            return _torch.tensor(False, dtype=_torch.bool)

        if isinstance(basket_xy, _np.ndarray) and basket_xy.ndim == 2:
            cur_basket_xy = basket_xy.mean(axis=0)
            cur_basket_z = float(_np.asarray(basket_z).mean())
            cur_milk_xy = (
                milk_xy.mean(axis=0)
                if isinstance(milk_xy, _np.ndarray) and milk_xy.ndim == 2
                else _np.asarray(milk_xy)
            )
            cur_milk_z = float(_np.asarray(milk_z).mean())
        else:
            cur_basket_xy = _np.asarray(basket_xy)
            cur_basket_z = float(basket_z)
            cur_milk_xy = _np.asarray(milk_xy)
            cur_milk_z = float(milk_z)

        try:
            cur_x = float(cur_basket_xy[0])
        except Exception:
            cur_x = float(cur_basket_xy)
        try:
            cur_y = float(cur_basket_xy[1]) if len(cur_basket_xy) > 1 else 0.0
        except Exception:
            cur_y = 0.0

        if self._hb_orig_basket_x is None and self._hb_diag_step >= 1:
            self._hb_orig_basket_x = cur_x
        if self._hb_orig_basket_y is None:
            self._hb_orig_basket_y = cur_y
        if self._hb_orig_basket_z is None and self._hb_diag_step >= 1:
            self._hb_orig_basket_z = cur_basket_z

        IN_BASKET_DIST = 0.10
        MOVE_Y_DELTA = 0.15
        Z_LIFT_DELTA = 0.01
        REQUIRED_STABLE_STEPS = 75

        dist = float(_np.linalg.norm(cur_milk_xy - cur_basket_xy))
        milk_above_basket = cur_milk_z > cur_basket_z
        in_basket = dist < IN_BASKET_DIST and milk_above_basket

        x_disp = None
        picked = False
        moved_left = False
        if getattr(self, '_hb_orig_basket_y', None) is not None:
            try:
                y_disp = float(cur_y) - float(self._hb_orig_basket_y)
            except Exception:
                y_disp = None
            if getattr(self, '_hb_orig_basket_x', None) is not None:
                try:
                    x_disp = float(self._hb_orig_basket_x) - cur_x
                except Exception:
                    x_disp = None
            if self._hb_orig_basket_z is not None:
                picked = (cur_basket_z - float(self._hb_orig_basket_z)) > Z_LIFT_DELTA
            moved_left = (y_disp is not None and float(y_disp) > MOVE_Y_DELTA) and (picked or in_basket)

        try:
            current_env_step = int(_np.asarray(self._elapsed_steps.detach().cpu()).reshape(-1)[0])
        except Exception:
            current_env_step = None

        last_env_step = getattr(self, "_hb_last_success_check_env_step", None)
        if moved_left and in_basket:
            if current_env_step is None or last_env_step is None:
                elapsed_env_steps = 1
            else:
                elapsed_env_steps = max(1, int(current_env_step) - int(last_env_step))
            self._hb_stable_steps += elapsed_env_steps
        else:
            self._hb_stable_steps = 0
        self._hb_last_success_check_env_step = current_env_step

        success = self._hb_stable_steps >= REQUIRED_STABLE_STEPS

        check_count = int(getattr(self, "_hb_debug_check_count", 0)) + 1
        self._hb_debug_check_count = check_count

        self._hb_debug_flags = {
            "success": bool(success),
            "in_basket": bool(in_basket),
            "milk_above_basket": bool(milk_above_basket),
            "milk_basket_dist": float(dist),
            "moved_left": bool(moved_left),
            "picked": bool(picked),
            "stable_steps": int(self._hb_stable_steps),
            "required_stable_steps": int(REQUIRED_STABLE_STEPS),
            "x_disp": None if x_disp is None else float(x_disp),
            "y_disp": None if 'y_disp' not in locals() or y_disp is None else float(y_disp),
            "move_y_delta": float(MOVE_Y_DELTA),
            "basket_x": float(cur_x),
            "basket_y": float(cur_y),
            "basket_z": float(cur_basket_z),
            "milk_z": float(cur_milk_z),
            "orig_basket_x": self._hb_orig_basket_x,
            "orig_basket_y": self._hb_orig_basket_y,
            "orig_basket_z": self._hb_orig_basket_z,
            "current_env_step": current_env_step,
            "check_count": check_count,
        }

        if str(os.environ.get("ROBOSYN_DEBUG_SUCCESS_FLAGS", "")).lower() in {"1", "true", "yes", "on"}:
            signature = (
                self._hb_debug_flags["in_basket"],
                self._hb_debug_flags["moved_left"],
                self._hb_debug_flags["picked"],
                self._hb_debug_flags["stable_steps"],
                self._hb_debug_flags["success"],
            )
            if signature != getattr(self, "_hb_debug_last_signature", None) or check_count % 10 == 0:
                self._hb_debug_last_signature = signature
                print(f"[HandleBasket success flags] {self._hb_debug_flags}")

        self._hb_diag_step += 1

        return _torch.tensor(success, dtype=_torch.bool)

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
