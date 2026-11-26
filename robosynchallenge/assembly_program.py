"""Qualify the Task Program's two-arm docking with measured tube contact."""

import math

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env

from .task_program_adapter import compose_task_program

_DEPLOYMENT, _ADAPTER = compose_task_program("item_assembly", ())


@register_env(
    "RoboSyn-ItemAssembly-v1",
    max_episode_steps=1500,
    task_program_registration=_ADAPTER.registration,
    task_program_adapter_factory=_ADAPTER,
)
class ItemAssembly(EmbodiedEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.task_program is None:
            cfg.task_program = _DEPLOYMENT.load_program()
        super().__init__(cfg=cfg, **kwargs)
        self._sleeve = self.sim.get_rigid_object("guijiao1")
        self._insert = self.sim.get_rigid_object("guijiao2")
        self._contact_sensor = self.sim.get_sensor("guijiao_contact")
        self._sleeve_id = self._contact_sensor.get_actor_ids("guijiao1")
        self._insert_id = self._contact_sensor.get_actor_ids("guijiao2")

    def _docking_metrics(self):
        sleeve = self._sleeve.get_local_pose(to_matrix=True)
        insert = self._insert.get_local_pose(to_matrix=True)
        sleeve_axis = sleeve[:, :3, 0]
        alignment = (sleeve_axis * insert[:, :3, 0]).sum(dim=-1).abs().clamp_max(1)
        angle = torch.acos(alignment)
        delta = insert[:, :3, 3] - sleeve[:, :3, 3]
        along_axis = (delta * sleeve_axis).sum(dim=-1, keepdim=True) * sleeve_axis
        lateral = torch.linalg.vector_norm(delta - along_axis, dim=-1)

        contacts = self._contact_sensor.get_data()
        a, b = contacts["user_ids"].unbind(dim=-1)
        pair = ((a == self._sleeve_id) & (b == self._insert_id)) | (
            (a == self._insert_id) & (b == self._sleeve_id)
        )
        minimum = (
            contacts["distance"]
            .masked_fill(~(pair & contacts["is_valid"]), torch.inf)
            .min(dim=1)
            .values
        )
        return angle, lateral, minimum

    def compute_task_state(self, **kwargs):
        angle, lateral, separation = self._docking_metrics()
        keep_running = torch.zeros_like(angle, dtype=torch.bool)
        return (
            keep_running,
            keep_running,
            {
                "tube_angle_rad": angle,
                "tube_lateral_offset_m": lateral,
                "tube_contact_separation_m": separation,
            },
        )

    def is_task_success(self, **kwargs):
        angle, lateral, separation = self._docking_metrics()
        return (
            (angle <= math.radians(15))
            & (lateral <= 0.02)
            & (separation <= 0.003)
            & (self._elapsed_steps >= 5)
        )
