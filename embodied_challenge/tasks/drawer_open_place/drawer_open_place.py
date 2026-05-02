import torch
from typing import Dict, Optional

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from embodichain.utils import logger

from .action_bank import DrawerOpenPlaceActionBank

__all__ = ["DrawerOpenPlaceEnv", "DrawerOpenPlaceAgentEnv"]


@register_env("DrawerOpenPlace-v1", max_episode_steps=900)
class DrawerOpenPlaceEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def create_demo_action_list(self, *args, **kwargs):
        logger.log_info("Create demo action list for DrawerOpenPlace task.")

        if getattr(self, "action_config", None) is None:
            logger.log_error("No action_config found in env, please check again.")
            return None

        self._init_action_bank(DrawerOpenPlaceActionBank, self.action_config)
        action_list = self.create_expert_demo_action_list(*args, **kwargs)
        if action_list is None:
            return None

        logger.log_info(
            f"Demo action list created with {len(action_list)} steps.", color="green"
        )
        return action_list

    def create_expert_demo_action_list(self, **kwargs):
        if hasattr(self, "action_bank") is False or self.action_bank is None:
            logger.log_error(
                "Action bank is not initialized. Cannot create expert demo action list."
            )
            return None

        ret = self.action_bank.create_action_list(
            self, self.graph_compose, self.packages
        )
        if ret is None:
            logger.log_warning("Failed to generate expert demo action list.")
            return None

        left_arm_joints = self.robot.get_joint_ids(name="left_arm", remove_mimic=True)
        right_arm_joints = self.robot.get_joint_ids(name="right_arm", remove_mimic=True)
        left_eef_joints = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
        right_eef_joints = self.robot.get_joint_ids(name="right_eef", remove_mimic=True)

        total_traj_num = ret[list(ret.keys())[0]].shape[-1]
        num_active_joints = len(self.active_joint_ids)
        actions = torch.zeros(
            (total_traj_num, self.num_envs, num_active_joints), dtype=torch.float32
        )

        global_to_active_idx = {
            joint_id: active_idx
            for active_idx, joint_id in enumerate(self.active_joint_ids)
        }

        for key, joints in [
            ("left_arm", left_arm_joints),
            ("left_eef", left_eef_joints),
            ("right_arm", right_arm_joints),
            ("right_eef", right_eef_joints),
        ]:
            if key not in ret:
                continue
            local_action_data = torch.as_tensor(ret[key].T, dtype=torch.float32)
            for i, joint_id in enumerate(joints):
                if joint_id in global_to_active_idx:
                    active_idx = global_to_active_idx[joint_id]
                    actions[:, 0, active_idx] = local_action_data[:, i]
        return actions

    def is_task_success(self, **kwargs) -> torch.Tensor:
        duck = self.sim.get_rigid_object("duck")
        drawer = self.sim.get_articulation("drawer")

        duck_final_xpos = duck.get_local_pose(to_matrix=True)
        drawer_pose = drawer.get_link_pose("outer_box", to_matrix=True)

        duck_pos_xy = duck_final_xpos[:, :2, 3]
        drawer_pos_xy = drawer_pose[:, :2, 3]

        # Success requires the duck to stay near the drawer in XY plane.
        duck_drawer_dist = torch.linalg.norm(duck_pos_xy - drawer_pos_xy, dim=-1)
        print(f"Duck-Drawer distance: {duck_drawer_dist.item():.4f}")
        dist_threshold = 0.1
        duck_near_drawer = duck_drawer_dist <= dist_threshold

        return duck_near_drawer


@register_env("DrawerOpenPlaceAgent-v1", max_episode_steps=900)
class DrawerOpenPlaceAgentEnv(BaseAgentEnv, DrawerOpenPlaceEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info
