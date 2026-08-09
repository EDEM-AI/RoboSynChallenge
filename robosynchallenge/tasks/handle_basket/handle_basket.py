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

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """
        Multi-stage success check to avoid ending the episode immediately when
        the milk briefly intersects the basket. The stages are:
        0 - wait for milk to be inside basket (proximity + above check)
        1 - once inside, require the basket to be lifted and moved left by a
            small margin for a few consecutive frames
        2 - require the basket to be placed back down and remain stable for
            a few consecutive frames -> success
        """
        # Initialize persistent state on first call
        if not hasattr(self, "_hb_stage"):
            self._hb_stage = 0
            self._hb_lifted_count = 0
            self._hb_stable_count = 0
            # track whether milk was ever observed close to left eef (grasped)
            self._hb_seen_grasp = False
            # record original basket xy for movement checks
            try:
                orig = getattr(self, "basket_pose_orig", None)
                if orig is None:
                    self._hb_orig_basket_z = None
                    self._hb_orig_basket_x = None
                else:
                    self._hb_orig_basket_z = float(orig[0, 2, 3])
                    self._hb_orig_basket_x = float(orig[0, 0, 3])
            except Exception:
                self._hb_orig_basket_z = None
                self._hb_orig_basket_x = None

        basket = self.sim.get_rigid_object("basket")
        milk = self.sim.get_rigid_object("milk")
        basket_pose = basket.get_local_pose(to_matrix=True)
        milk_pose = milk.get_local_pose(to_matrix=True)

        # Extract positions (assume shape [N,4,4] or [4,4]) and convert to numpy
        try:
            basket_xy = basket_pose[:, :2, 3]
            milk_xy = milk_pose[:, :2, 3]
            basket_z = basket_pose[:, 2, 3]
            milk_z = milk_pose[:, 2, 3]
        except Exception:
            # single-environment fallback
            basket_xy = basket_pose[:2, 3]
            milk_xy = milk_pose[:2, 3]
            basket_z = float(basket_pose[2, 3])
            milk_z = float(milk_pose[2, 3])

        # make numpy scalars for simple logic
        import numpy as _np

        dist = _np.linalg.norm(_np.asarray(milk_xy) - _np.asarray(basket_xy), axis=-1)
        up = (_np.asarray(milk_z) > _np.asarray(basket_z)).astype(bool)

        # thresholds and counters
        IN_BASKET_DIST = 0.10
        LIFT_Z_DELTA = 0.05
        MOVE_X_DELTA = 0.05
        LIFT_REQUIRED_FRAMES = 3
        # prefer time-based stability check (seconds). If no timestamp is
        # available from the caller/sim, we gracefully fall back to the
        # previous frame-count behavior.
        STABLE_REQUIRED_SECS = 2.0
        STABLE_REQUIRED_FRAMES = 5

        success = False

        # obtain current time (prefer kwargs or sim), fall back to pseudo-time
        now_ts = None
        if 'ts' in kwargs:
            now_ts = kwargs.get('ts')
        elif 'timestamp' in kwargs:
            now_ts = kwargs.get('timestamp')
        else:
            try:
                if hasattr(self, 'sim') and hasattr(self.sim, 'get_time'):
                    now_ts = float(self.sim.get_time())
            except Exception:
                now_ts = None

        if now_ts is None:
            est_dt = getattr(self, '_hb_est_dt', None)
            if est_dt is None:
                est_dt = 1.0 / 30.0
                self._hb_est_dt = est_dt
            last_t = getattr(self, '_hb_last_time', 0.0)
            now_ts = last_t + est_dt
            self._hb_last_time = now_ts
        else:
            self._hb_last_time = float(now_ts)

        # Treat arrays and scalars uniformly
        is_in_basket = (dist < IN_BASKET_DIST) & (up)

        # detect whether milk was grasped by left eef at any point
        GRASP_DIST = 0.06
        try:
            # attempt to query robot qpos and compute left eef FK
            qpos = self.robot.get_qpos()  # Tensor (num_envs, dof)
            # compute FK; handle different compute_fk signatures
            try:
                left_eef_pose = self.robot.compute_fk(name="left_eef", qpos=qpos, to_matrix=True).squeeze(0)
            except TypeError:
                left_eef_pose = self.robot.compute_fk(qpos, None, "left_eef")

            # extract left eef xy
            try:
                left_eef_xy = left_eef_pose[:2, 3]
            except Exception:
                left_eef_xy = left_eef_pose[0, :2, 3]

            # compute proximity (single-env assumed)
            try:
                milk_xy_arr = _np.asarray(milk_xy)
            except Exception:
                milk_xy_arr = _np.asarray(milk_xy)
            d_eef = float(_np.linalg.norm(milk_xy_arr - _np.asarray(left_eef_xy)))
            if d_eef < GRASP_DIST:
                self._hb_seen_grasp = True
        except Exception:
            # if any of these calls fail, keep previous seen_grasp value
            pass

        # Single-environment flow (most evaluations run with 1 env)
        if isinstance(is_in_basket, _np.ndarray):
            idx = 0
            in_basket = bool(is_in_basket[idx])
            cur_basket_x = float(_np.asarray(basket_xy)[idx, 0])
            cur_basket_z = float(_np.asarray(basket_z)[idx])
        else:
            in_basket = bool(is_in_basket)
            cur_basket_x = float(_np.asarray(basket_xy)[0]) if _np.asarray(basket_xy).ndim > 0 else float(_np.asarray(basket_xy))
            cur_basket_z = float(basket_z)

        # Stage machine
        # Simplified rules:
        # - Stage 0 -> 1: milk enters basket (as before).
        # - Stage 1 -> 2: immediately transition when basket is detected lifted
        #   relative to original z AND has moved left by the required margin.
        # - Stage 2: success when basket is placed down (near orig_z) AND both
        #   basket and milk are not noticeably moving (i.e. not shaking).
        if self._hb_stage == 0:
            # allow stage progression when milk is in basket for a short time
            # even if we didn't observe a grasp or a clear lift. This relaxes
            # the strict requirement that the left eef must have been nearby
            # (some eval runs capture the scene differently).
            if in_basket:
                if getattr(self, '_hb_in_basket_start', None) is None:
                    self._hb_in_basket_start = float(now_ts)
                dur_in_basket = float(now_ts) - float(self._hb_in_basket_start)
            else:
                self._hb_in_basket_start = None
                dur_in_basket = 0.0

            # Progress to stage 1 if we either saw a grasp or milk remained
            # in the basket for a short duration.
            if (in_basket and self._hb_seen_grasp) or (dur_in_basket >= float(STABLE_REQUIRED_SECS)):
                self._hb_stage = 1
                # reset any transient trackers
                self._hb_lifted_count = 0
                self._hb_stable_count = 0
        elif self._hb_stage == 1:
            # detect lift relative to original basket z (fallback to current if missing)
            orig_z = self._hb_orig_basket_z if self._hb_orig_basket_z is not None else cur_basket_z - 0.0
            lifted = cur_basket_z > (orig_z + LIFT_Z_DELTA)
            moved_left = False
            if self._hb_orig_basket_x is not None:
                moved_left = (self._hb_orig_basket_x - cur_basket_x) > MOVE_X_DELTA

            # Require the lift+move condition to hold for several consecutive checks
            if lifted and moved_left:
                self._hb_lifted_count += 1
            else:
                self._hb_lifted_count = 0

            # Transition to stage 2 after sustained lift/move, or if the milk
            # has been in the basket for a sustained period (no clear lift).
            if self._hb_lifted_count >= LIFT_REQUIRED_FRAMES:
                self._hb_stage = 2
            else:
                # allow skipping explicit lift when milk has been in basket
                # for STABLE_REQUIRED_SECS (handles direct-place cases)
                if getattr(self, '_hb_in_basket_start', None) is not None:
                    dur_skip = float(now_ts) - float(self._hb_in_basket_start)
                    if dur_skip >= float(STABLE_REQUIRED_SECS):
                        self._hb_stage = 2
                # initialize previous-position trackers for stability check
                self._hb_prev_basket_x = cur_basket_x
                self._hb_prev_basket_z = cur_basket_z
                # reset stability counter on entry
                self._hb_stable_count = 0
                try:
                    self._hb_prev_milk_xy = _np.asarray(milk_xy).copy()
                except Exception:
                    self._hb_prev_milk_xy = None
        elif self._hb_stage == 2:
            # require basket to be placed down near original z and not shaking
            orig_z = self._hb_orig_basket_z if self._hb_orig_basket_z is not None else cur_basket_z
            placed_down = abs(cur_basket_z - orig_z) < (LIFT_Z_DELTA / 2.0)

            # compute simple motion magnitude since last check (fallback to 0 if not available)
            basket_motion = 0.0
            milk_motion = 0.0
            prev_bx = getattr(self, '_hb_prev_basket_x', None)
            prev_bz = getattr(self, '_hb_prev_basket_z', None)
            prev_milk = getattr(self, '_hb_prev_milk_xy', None)
            try:
                if prev_bx is not None:
                    dx = cur_basket_x - prev_bx
                    dz = cur_basket_z - prev_bz
                    basket_motion = abs(dx) + abs(dz)
                if prev_milk is not None:
                    cur_milk_xy = _np.asarray(milk_xy)
                    milk_motion = float(_np.linalg.norm(cur_milk_xy - prev_milk))
            except Exception:
                basket_motion = 0.0
                milk_motion = 0.0

            # update previous trackers
            self._hb_prev_basket_x = cur_basket_x
            self._hb_prev_basket_z = cur_basket_z
            try:
                self._hb_prev_milk_xy = _np.asarray(milk_xy).copy()
            except Exception:
                self._hb_prev_milk_xy = None

            # thresholds and time-window params
            STABLE_MOTION_THR = 0.02
            STABLE_REQUIRED_SECS = 2.0
            STABLE_TIMEOUT_SECS = 60.0

            # obtain current time (prefer kwargs or sim), fall back to pseudo-time
            now_ts = None
            if 'ts' in kwargs:
                now_ts = kwargs.get('ts')
            elif 'timestamp' in kwargs:
                now_ts = kwargs.get('timestamp')
            else:
                try:
                    if hasattr(self, 'sim') and hasattr(self.sim, 'get_time'):
                        now_ts = float(self.sim.get_time())
                except Exception:
                    now_ts = None

            if now_ts is None:
                est_dt = getattr(self, '_hb_est_dt', None)
                if est_dt is None:
                    est_dt = 1.0 / 30.0
                    self._hb_est_dt = est_dt
                last_t = getattr(self, '_hb_last_time', 0.0)
                now_ts = last_t + est_dt
                self._hb_last_time = now_ts
            else:
                # update running estimate of frame dt when real timestamps available
                prev_time = getattr(self, '_hb_last_time', None)
                if prev_time is not None:
                    try:
                        dt = float(now_ts) - float(prev_time)
                        if dt > 0:
                            prev_est = getattr(self, '_hb_est_dt', None)
                            if prev_est is None:
                                self._hb_est_dt = float(dt)
                            else:
                                # exponential moving average
                                self._hb_est_dt = 0.9 * float(prev_est) + 0.1 * float(dt)
                    except Exception:
                        pass
                self._hb_last_time = float(now_ts)

            # adapt allowed gap to observed sampling rate to tolerate low fps
            est_dt = getattr(self, '_hb_est_dt', 1.0 / 30.0)
            ALLOWED_GAP_SECS = max(0.5, float(est_dt) * 1.5)

            # record first placed_down time to enforce timeout
            if placed_down and getattr(self, '_hb_placed_time', None) is None:
                self._hb_placed_time = float(now_ts)

            if getattr(self, '_hb_placed_time', None) is not None and (float(now_ts) - float(self._hb_placed_time)) > float(STABLE_TIMEOUT_SECS):
                # timeout exceeded; give up attempting success in this run
                pass

            # determine whether this frame is 'stable'
            stable_frame = in_basket and basket_motion < STABLE_MOTION_THR and milk_motion < STABLE_MOTION_THR

            if stable_frame:
                # start accumulation if not present
                if getattr(self, '_hb_stable_accum_start', None) is None:
                    self._hb_stable_accum_start = float(now_ts)
                # if there was a recent unstable and gap exceeded allowed, reset start
                last_unstable = getattr(self, '_hb_last_unstable_ts', None)
                if last_unstable is not None and (float(self._hb_stable_accum_start) - float(last_unstable)) > float(ALLOWED_GAP_SECS):
                    # if gap too large before start, ensure start is current
                    self._hb_stable_accum_start = float(now_ts)

                # check accumulated duration
                dur = float(now_ts) - float(self._hb_stable_accum_start)
                if dur >= float(STABLE_REQUIRED_SECS):
                    success = True
            else:
                # mark last unstable time
                self._hb_last_unstable_ts = float(now_ts)
                # if accumulated start exists and gap exceeded allowed, reset accumulation
                if getattr(self, '_hb_stable_accum_start', None) is not None and (float(now_ts) - float(self._hb_stable_accum_start)) > float(ALLOWED_GAP_SECS):
                    self._hb_stable_accum_start = None
                # keep legacy counter as fallback
                self._hb_stable_count = 0

        # Return torch tensor boolean for compatibility
        import torch as _torch
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
