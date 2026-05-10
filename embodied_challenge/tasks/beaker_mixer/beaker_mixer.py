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
from typing import Dict, Optional, Sequence

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodied_challenge.managers.events import visualize_affordance_pose
from embodichain.utils import logger

from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from .action_bank import BeakerMixerActionBank

__all__ = ["BeakerMixerEnv", "BeakerMixerTestEnv", "BeakerMixerAgentEnv"]


@register_env("BeakerMixer-v0", max_episode_steps=600)
class BeakerMixerEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        debug_render_cfg = {}
        if isinstance(action_config, dict):
            debug_render_cfg = action_config.pop("debug_render", {})

        if action_config is not None:
            self.action_config = action_config

        self._button_contact_happened = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._button_region_radius = 0.01
        self._button_impulse_threshold = 0.01
        self._button_contact_sensor = None
        self._arm_link_user_ids = torch.empty(
            0, dtype=torch.int32, device=self.device
        )
        self._mixer_user_ids = torch.empty(0, dtype=torch.int32, device=self.device)

        self._init_button_contact_sensor()

    def _init_button_contact_sensor(self) -> None:
        self._button_contact_sensor = self.sim.get_sensor("beaker_mixer_button_contact")
        if self._button_contact_sensor is None:
            logger.log_warning(
                "Contact sensor 'beaker_mixer_button_contact' not found in config; button-force success check is disabled."
            )
            return
        self._button_contact_sensor.set_contact_point_visibility(
            visible=True,
            rgba=(0.0, 0.0, 1.0, 1.0),  # Blue color
            point_size=10.0,
        )

        self._mixer_user_ids = self.sim.get_rigid_object("beaker_mixer").get_user_ids()
        self._arm_link_user_ids = self._collect_arm_link_user_ids(
            ["left_eef"]
        )
        if self._arm_link_user_ids.numel() == 0:
            self._arm_link_user_ids = self.robot.get_user_ids().reshape(-1).to(
                dtype=torch.int32
            )

    def _collect_arm_link_user_ids(self, part_names: Sequence[str]) -> torch.Tensor:
        user_ids = []
        for part_name in part_names:
            try:
                link_names = self.robot.get_link_names(name=part_name)
                print(f"link_names: {link_names}")
            except Exception:
                link_names = None
            if not link_names:
                continue
            for link_name in link_names:
                user_ids.append(self.robot.get_user_ids(link_name=link_name).reshape(-1))

        if len(user_ids) == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        return torch.unique(torch.cat(user_ids, dim=0)).to(dtype=torch.int32)

    def _get_scalar_from_affordance(self, keys: Sequence[str], default: float) -> float:
        for key in keys:
            value = self.affordance_datas.get(key, None)
            if value is None:
                continue
            if torch.is_tensor(value):
                return float(value.reshape(-1)[0].item())
            return float(value)
        return default

    def _get_button_position(self, mixer_pose: torch.Tensor) -> torch.Tensor:
        button_pose = self._get_button_pose(mixer_pose)
        return button_pose[:, :3, 3]

    def _get_button_pose(self, mixer_pose: torch.Tensor) -> torch.Tensor:
        offset_x = self._get_scalar_from_affordance(
            ["beaker_mixer_button_offset_x", "button_offset_x"], 0.11175
        )
        offset_y = self._get_scalar_from_affordance(
            ["beaker_mixer_button_offset_y", "button_offset_y"], -0.006
        )
        offset_z = self._get_scalar_from_affordance(
            ["beaker_mixer_button_offset_z", "button_offset_z"], 0.042
        )

        local_button_pose = torch.eye(
            4, dtype=mixer_pose.dtype, device=mixer_pose.device
        ).view(1, 4, 4)
        local_button_pose = local_button_pose.repeat(mixer_pose.shape[0], 1, 1)
        local_button_pose[:, 0, 3] = offset_x
        local_button_pose[:, 1, 3] = offset_y
        local_button_pose[:, 2, 3] = offset_z

        return torch.bmm(mixer_pose, local_button_pose)

    def _visualize_button_axis(self, mixer_pose: torch.Tensor) -> None:
        if self.sim_cfg.headless:
            return

        self.affordance_datas["beaker_mixer_button_pose"] = self._get_button_pose(
            mixer_pose
        )
        visualize_affordance_pose(
            env=self,
            env_ids=None,
            pose_key="beaker_mixer_button_pose",
            marker_name="beaker_mixer_button_axis",
            axis_size=0.003,
            axis_len=0.05,
            arena_index=0,
            remove_old=True,
        )

    def _update_button_contact_history(self) -> None:
        if self._button_contact_sensor is None:
            return

        contact_report = self._button_contact_sensor.get_data()
        valid_mask = contact_report["is_valid"]
        if not valid_mask.any():
            return

        contact_user_ids = contact_report["user_ids"][valid_mask]
        contact_impulse = contact_report["impulse"][valid_mask]
        contact_position = contact_report["position"][valid_mask]

        mixer_contact = torch.isin(
            contact_user_ids[..., 0], self._mixer_user_ids
        ) | torch.isin(contact_user_ids[..., 1], self._mixer_user_ids)
        arm_contact = torch.isin(
            contact_user_ids[..., 0], self._arm_link_user_ids
        ) | torch.isin(contact_user_ids[..., 1], self._arm_link_user_ids)
        # print(f"arm_contact:{arm_contact}")
        # print(f"self._mixer_user_ids:{self._mixer_user_ids}")
        # print(f"self._arm_link_user_ids:{self._arm_link_user_ids}")
        # print(f"contact_user_ids_0:{contact_user_ids[...,0]}, 1:{contact_user_ids[...,1]}")
        mixer_pose = self.sim.get_rigid_object("beaker_mixer").get_local_pose(to_matrix=True)
        button_position = self._get_button_position(mixer_pose)
        button_dist = torch.linalg.norm(
            contact_position - button_position.unsqueeze(1), dim=-1
        )
        in_button_region = button_dist <= self._button_region_radius

        impulse_valid = contact_impulse >= self._button_impulse_threshold
        print(f"Contact impulses: {contact_impulse}")
        print(f"Contact positions: {contact_position}")
        print(f"Button position: {button_position}")
        print(f"in_button_region: {in_button_region}")
        print(f"impulse_valid: {impulse_valid}")
        press_mask = mixer_contact & arm_contact & in_button_region & impulse_valid

        self._button_contact_happened |= press_mask.any(dim=1)

    def create_demo_action_list(self, *args, **kwargs):
        """Create a demonstration action list for BeakerMixer task."""
        logger.log_info("Create demo action list for BeakerMixerTask.")
        action_list = None

        if getattr(self, "action_config") is not None:
            self._init_action_bank(BeakerMixerActionBank, self.action_config)
            action_list = self.create_expert_demo_action_list(*args, **kwargs)
        else:
            logger.log_error("No action_config found in env, please check again.")

        if action_list is None:
            return action_list

        logger.log_info(
            f"Demo action list created with {len(action_list)} steps.", color="green"
        )
        return action_list

    def create_expert_demo_action_list(self, **kwargs):
        """Create expert demonstration actions from action bank graph."""
        if hasattr(self, "action_bank") is False or self.action_bank is None:
            logger.log_error(
                "Action bank is not initialized. Cannot create expert demo action list."
            )

        ret = self.action_bank.create_action_list(
            self, self.graph_compose, self.packages
        )

        if ret is None:
            logger.log_warning("Failed to generate expert demo action list.")
            return None

        left_arm_joints = self.robot.get_joint_ids(name="left_arm", remove_mimic=True)
        right_arm_joints = self.robot.get_joint_ids(
            name="right_arm", remove_mimic=True
        )
        left_eef_joints = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
        right_eef_joints = self.robot.get_joint_ids(
            name="right_eef", remove_mimic=True
        )

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
            if key in ret:
                # TODO: only 1 env supported now
                local_action_data = torch.as_tensor(ret[key].T, dtype=torch.float32)
                for i, joint_id in enumerate(joints):
                    if joint_id in global_to_active_idx:
                        active_idx = global_to_active_idx[joint_id]
                        actions[:, 0, active_idx] = local_action_data[:, i]
        return actions

    def compute_task_state(self, **kwargs):
        beaker = self.sim.get_rigid_object("beaker")
        mixer = self.sim.get_rigid_object("beaker_mixer")

        self._update_button_contact_history()

        beaker_pose = beaker.get_local_pose(to_matrix=True)
        mixer_pose = mixer.get_local_pose(to_matrix=True)
        # self._visualize_button_axis(mixer_pose)

        beaker_fall = self._is_fall(beaker_pose)
        success = torch.zeros_like(beaker_fall, dtype=torch.bool)

        return success, beaker_fall, {}

    def is_task_success(self, **kwargs) -> torch.Tensor:
        self._update_button_contact_history()

        beaker = self.sim.get_rigid_object("beaker")
        beaker_mixer = self.sim.get_rigid_object("beaker_mixer")

        beaker_final_xpos = beaker.get_local_pose(to_matrix=True)
        beaker_mixer_final_xpos = beaker_mixer.get_local_pose(to_matrix=True)

        beaker_ret = self._is_fall(beaker_final_xpos)
        beaker_pos_xy = beaker_final_xpos[:, :2, 3]
        beaker_mixer_pos_xy = beaker_mixer_final_xpos[:, :2, 3]

        # Success requires the beaker to stay near the mixer in XY plane.
        beaker_mixer_dist = torch.linalg.norm(beaker_pos_xy - beaker_mixer_pos_xy, dim=-1)
        print(f"Beaker-Mixer distance: {beaker_mixer_dist.item():.4f}")
        dist_threshold = 0.08
        beaker_near_mixer = beaker_mixer_dist <= dist_threshold
        print(f"beaker_near_mixer:{beaker_near_mixer}, _button_contact_happened:{self._button_contact_happened}, beaker_ret:{beaker_ret}")
        return (~beaker_ret) & beaker_near_mixer & self._button_contact_happened

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)

        if options is None:
            options = {}
        reset_ids = options.get(
            "reset_ids",
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device),
        )
        self._button_contact_happened[reset_ids] = False

        return obs, info

    def _is_fall(self, pose: torch.Tensor) -> torch.Tensor:
        # Extract z-axis from rotation matrix (last column, first 3 elements)
        pose_rz = pose[:, :3, 2]
        world_z_axis = torch.tensor([0, 0, 1], dtype=pose.dtype, device=pose.device)

        # Compute dot product for each batch element
        dot_product = torch.sum(pose_rz * world_z_axis, dim=-1)  # Shape: (batch_size,)

        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Compute angle and check if fallen
        angle = torch.arccos(dot_product)
        return angle >= torch.pi / 3


@register_env("BeakerMixerTest-v0", max_episode_steps=600)
class BeakerMixerTestEnv(BeakerMixerEnv):
    def compute_task_state(self, **kwargs):
        button = self.sim.get_articulation("button")
        button_qpos = button.get_qpos()

        # button.urdf uses a single prismatic joint with range [-0.005, 0.0].
        # Treat any detectable displacement as success (with tiny epsilon to avoid numerical noise).
        press_depth = -button_qpos[:, 0]
        movement_threshold = 0.004
        success = press_depth >= movement_threshold
        # print(f"press_depth: {press_depth}, movement_threshold: {movement_threshold}")
        self._button_pressed |= success
        fail = torch.zeros_like(success, dtype=torch.bool)

        return success, fail, {}

@register_env("BeakerMixerAgent-v0", max_episode_steps=600)
class BeakerMixerAgentEnv(BaseAgentEnv, BeakerMixerEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info