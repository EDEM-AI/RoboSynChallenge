# # ----------------------------------------------------------------------------
# # Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ----------------------------------------------------------------------------

# import torch
# from typing import Dict, Optional, Sequence

# from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
# from embodichain.lab.gym.utils.registration import register_env
# from embodichain.utils import logger
# from embodichain.lab.sim.cfg import MarkerCfg

# from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
# from .action_bank import (
#     BeatHammerBlockActionBank,
# )

# __all__ = ["BeatHammerBlockEnv", "BeatHammerBlockTestEnv", "BeatHammerBlockAgentEnv"]



# @register_env("BeatHammerBlock-v3", max_episode_steps=600)
# class BeatHammerBlockEnv(EmbodiedEnv):

#     def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
#         super().__init__(cfg, **kwargs)

#         action_config = kwargs.get("action_config", None)
#         if action_config is not None:
#             self.action_config = action_config

#         self._button_pressed = torch.zeros(
#             self.num_envs, dtype=torch.bool, device=self.device
#         )
#         self._table_contact_debug_cache = None
#         self._table_contact_debug_warned = False

#     def _hook_after_sim_step(
#         self,
#         obs,
#         action,
#         rewards: torch.Tensor,
#         dones: torch.Tensor,
#         info: Dict,
#         **kwargs,
#     ):
#         super()._hook_after_sim_step(
#             obs=obs,
#             action=action,
#             rewards=rewards,
#             dones=dones,
#             info=info,
#             **kwargs,
#         )
#         if getattr(self, "debug_print_table_contact", False):
#             self._print_table_contact_debug(obs)

#     def _get_table_contact_debug_cache(self):
#         if self._table_contact_debug_cache is not None:
#             return self._table_contact_debug_cache

#         sensor_uid = getattr(self, "debug_table_contact_sensor_uid", "table_contact")
#         sensor = self.sensors.get(sensor_uid)
#         if sensor is None:
#             if not self._table_contact_debug_warned:
#                 logger.log_warning(
#                     f"Contact sensor '{sensor_uid}' not found; skip table contact debug."
#                 )
#                 self._table_contact_debug_warned = True
#             return None

#         cfg = getattr(sensor, "cfg", None)
#         empty_ids = torch.empty(0, dtype=torch.int32, device=self.device)
#         table_user_ids = []
#         left_user_ids = []
#         right_user_ids = []
#         left_link_names = []
#         right_link_names = []

#         for rigid_uid in getattr(cfg, "rigid_uid_list", []):
#             rigid_object = self.sim.get_rigid_object(rigid_uid)
#             if rigid_object is not None:
#                 table_user_ids.append(
#                     rigid_object.get_user_ids().reshape(-1).to(dtype=torch.int32)
#                 )

#         for articulation_cfg in getattr(cfg, "articulation_cfg_list", []):
#             articulation_uid = getattr(articulation_cfg, "articulation_uid", "")
#             articulation = self.sim.get_articulation(articulation_uid)
#             if articulation is None:
#                 articulation = self.sim.get_robot(articulation_uid)
#             if articulation is None:
#                 continue

#             link_names = (
#                 getattr(articulation_cfg, "link_name_list", [])
#                 or articulation.link_names
#             )
#             for link_name in link_names:
#                 link_name_lower = link_name.lower()
#                 link_user_ids = (
#                     articulation.get_user_ids(link_name=link_name)
#                     .reshape(-1)
#                     .to(dtype=torch.int32)
#                 )
#                 if "left" in link_name_lower:
#                     left_user_ids.append(link_user_ids)
#                     left_link_names.append(link_name)
#                 elif "right" in link_name_lower:
#                     right_user_ids.append(link_user_ids)
#                     right_link_names.append(link_name)

#         self._table_contact_debug_cache = {
#             "sensor_uid": sensor_uid,
#             "table_user_ids": (
#                 torch.unique(torch.cat(table_user_ids)) if table_user_ids else empty_ids
#             ),
#             "left_user_ids": (
#                 torch.unique(torch.cat(left_user_ids)) if left_user_ids else empty_ids
#             ),
#             "right_user_ids": (
#                 torch.unique(torch.cat(right_user_ids)) if right_user_ids else empty_ids
#             ),
#             "left_link_names": left_link_names,
#             "right_link_names": right_link_names,
#         }

#         if (
#             not self._table_contact_debug_warned
#             and (
#                 self._table_contact_debug_cache["table_user_ids"].numel() == 0
#                 or self._table_contact_debug_cache["left_user_ids"].numel() == 0
#                 or self._table_contact_debug_cache["right_user_ids"].numel() == 0
#             )
#         ):
#             logger.log_warning(
#                 "Incomplete table contact debug ids from contact sensor config; "
#                 f"sensor={sensor_uid}, "
#                 f"left_links={left_link_names}, right_links={right_link_names}."
#             )
#             self._table_contact_debug_warned = True

#         return self._table_contact_debug_cache

#     def _print_table_contact_debug(self, obs) -> None:
#         cache = self._get_table_contact_debug_cache()
#         if cache is None:
#             return

#         sensor_uid = cache["sensor_uid"]
#         sensor_obs = obs["sensor"] if "sensor" in obs.keys() else None
#         contact_report = None
#         if sensor_obs is not None and sensor_uid in sensor_obs.keys():
#             contact_report = sensor_obs[sensor_uid]
#         if contact_report is None:
#             sensor = self.sensors.get(sensor_uid)
#             if sensor is None:
#                 return
#             contact_report = sensor.get_data()

#         valid_mask = contact_report["is_valid"]
#         user_ids = contact_report["user_ids"]
#         impulse = contact_report["impulse"]

#         table_mask = torch.isin(
#             user_ids[..., 0], cache["table_user_ids"]
#         ) | torch.isin(user_ids[..., 1], cache["table_user_ids"])
#         left_mask = valid_mask & table_mask & (
#             torch.isin(user_ids[..., 0], cache["left_user_ids"])
#             | torch.isin(user_ids[..., 1], cache["left_user_ids"])
#         )
#         right_mask = valid_mask & table_mask & (
#             torch.isin(user_ids[..., 0], cache["right_user_ids"])
#             | torch.isin(user_ids[..., 1], cache["right_user_ids"])
#         )

#         left_counts = left_mask.sum(dim=1)
#         right_counts = right_mask.sum(dim=1)
#         left_max_impulse = torch.where(left_mask, impulse, torch.zeros_like(impulse))
#         right_max_impulse = torch.where(right_mask, impulse, torch.zeros_like(impulse))
#         left_max_impulse = left_max_impulse.max(dim=1).values
#         right_max_impulse = right_max_impulse.max(dim=1).values

#         step_ids = self._elapsed_steps.detach().cpu().tolist()
#         left_counts = left_counts.detach().cpu().tolist()
#         right_counts = right_counts.detach().cpu().tolist()
#         left_max_impulse = left_max_impulse.detach().cpu().tolist()
#         right_max_impulse = right_max_impulse.detach().cpu().tolist()

#         for env_id in range(self.num_envs):
#             left_count = int(left_counts[env_id])
#             right_count = int(right_counts[env_id])
#             print(
#                 f"[{sensor_uid}] step={int(step_ids[env_id])} env={env_id} "
#                 f"left_arm_table_collision={left_count > 0} "
#                 f"left_count={left_count} "
#                 f"left_max_impulse={float(left_max_impulse[env_id]):.6g} "
#                 f"right_arm_table_collision={right_count > 0} "
#                 f"right_count={right_count} "
#                 f"right_max_impulse={float(right_max_impulse[env_id]):.6g}",
#                 flush=True,
#             )

#     def create_demo_action_list(self, *args, **kwargs):
#         """
#         Create a demonstration action list for the current task.

#         Returns:
#             list: A list of demo actions generated by the task.
#         """
#         logger.log_info("Create demo action list for BeatHammerBlockTask.")

#         if getattr(self, "action_config") is not None:
#             self._init_action_bank(BeatHammerBlockActionBank, self.action_config)
#             action_list = self.create_expert_demo_action_list(*args, **kwargs)
#         else:
#             logger.log_error("No action_config found in env, please check again.")

#         if action_list is None:
#             return action_list

#         logger.log_info(
#             f"Demo action list created with {len(action_list)} steps.", color="green"
#         )
#         return action_list

#     def create_expert_demo_action_list(self, **kwargs):
#         """
#         Create an expert demonstration action list using the action bank.

#         This function generates a trajectory based on expert knowledge, mapping joint and end-effector
#         states to the required action format for the environment and robot type.

#         Args:
#             **kwargs: Additional keyword arguments.

#         Returns:
#             list: A list of actions, each containing joint positions ("qpos").
#         """

#         if hasattr(self, "action_bank") is False or self.action_bank is None:
#             logger.log_error(
#                 "Action bank is not initialized. Cannot create expert demo action list."
#             )

#         ret = self.action_bank.create_action_list(
#             self, self.graph_compose, self.packages
#         )

#         if ret is None:
#             logger.log_warning("Failed to generate expert demo action list.")
#             return None

#         # TODO: to be removed, need a unified interface in robot class
#         left_arm_joints = self.robot.get_joint_ids(name="left_arm", remove_mimic=True)
#         right_arm_joints = self.robot.get_joint_ids(name="right_arm", remove_mimic=True)
#         left_eef_joints = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
#         right_eef_joints = self.robot.get_joint_ids(name="right_eef", remove_mimic=True)


#         total_traj_num = ret[list(ret.keys())[0]].shape[-1]
#         num_active_joints = len(self.active_joint_ids)
#         actions = torch.zeros(
#             (total_traj_num, self.num_envs, num_active_joints), dtype=torch.float32
#         )

#         # 建立一个从全局 joint_id 到 active_joint_id 在 action 数组中正确存放位置的映射
#         global_to_active_idx = {
#             joint_id: active_idx for active_idx, joint_id in enumerate(self.active_joint_ids)
#         }

#         for key, joints in [
#             ("left_arm", left_arm_joints),
#             ("left_eef", left_eef_joints),
#             ("right_arm", right_arm_joints),
#             ("right_eef", right_eef_joints),
#         ]:
#             if key in ret:
#                 # TODO: only 1 env supported now
#                 local_action_data = torch.as_tensor(ret[key].T, dtype=torch.float32)

#                 # 【修改重点2】：使用映射精准定位它在 action tensor 中的正确位置存放
#                 for i, joint_id in enumerate(joints):
#                     if joint_id in global_to_active_idx:
#                         active_idx = global_to_active_idx[joint_id]
#                         actions[:, 0, active_idx] = local_action_data[:, i]
#         return actions
#     def compute_task_state(self, **kwargs):
#         button = self.sim.get_articulation("button")
#         button_qpos = button.get_qpos()

#         # button.urdf uses a single prismatic joint with range [-0.005, 0.0].
#         # Treat any detectable displacement as success (with tiny epsilon to avoid numerical noise).
#         press_depth = -button_qpos[:, 0]
#         movement_threshold = 0.0048
#         success = press_depth >= movement_threshold
#         # print(f"press_depth: {press_depth}, movement_threshold: {movement_threshold}")
#         self._button_pressed |= success
#         fail = torch.zeros_like(success, dtype=torch.bool)
#         success = torch.zeros_like(fail, dtype=torch.bool)
#         return success, fail, {}

#     def is_task_success(self, **kwargs) -> torch.Tensor:

#         return self._button_pressed
#     def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
#         obs, info = super().reset(seed=seed, options=options)

#         if options is None:
#             options = {}
#         reset_ids = options.get(
#             "reset_ids",
#             torch.arange(self.num_envs, dtype=torch.int32, device=self.device),
#         )
#         self._button_pressed[reset_ids] = False

#         return obs, info

# @register_env("BeatHammerBlockTest-v3", max_episode_steps=600)
# class BeatHammerBlockTestEnv(BeatHammerBlockEnv):
#     def compute_task_state(self, **kwargs):
#         button = self.sim.get_articulation("button")
#         button_qpos = button.get_qpos()

#         # button.urdf uses a single prismatic joint with range [-0.005, 0.0].
#         # Treat any detectable displacement as success (with tiny epsilon to avoid numerical noise).
#         press_depth = -button_qpos[:, 0]
#         movement_threshold = 0.004
#         success = press_depth >= movement_threshold
#         # print(f"press_depth: {press_depth}, movement_threshold: {movement_threshold}")
#         self._button_pressed |= success
#         fail = torch.zeros_like(success, dtype=torch.bool)

#         return success, fail, {}

# @register_env("BeatHammerBlockAgent-v3", max_episode_steps=600)
# class BeatHammerBlockAgentEnv(BaseAgentEnv, BeatHammerBlockEnv):
#     def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
#         super().__init__(cfg, **kwargs)
#         super()._init_agents(**kwargs)

#     def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
#         obs, info = super().reset(seed=seed, options=options)
#         super().get_states()
#         return obs, info
