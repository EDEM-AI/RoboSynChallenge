"""Check the pen's physical insertion after the native handover and placement."""

from dataclasses import replace
import math

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.task_program.language.loader import load_task_program

from .insertion_targets import (
    InsertionAlignmentFactory,
    InsertionPlaceFactory,
)
from .task_program_adapter import compose_task_program


def _pen_placement(release_height, *, integration_overrides=None, **geometry):
    placement = InsertionPlaceFactory(
        "pen",
        "holder",
        (0.0, release_height, 0.0),
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
        **geometry,
    )
    return compose_task_program(
        "items_handover",
        (placement,),
        additional_lowerers=(InsertionAlignmentFactory(placement),),
        use_scene_grasp_options=True,
        integration_overrides=integration_overrides,
    )


_DEPLOYMENT, _ADAPTER = _pen_placement(
    0.163,
    above_opening=0.020,
    integration_overrides={
        "profile": {
            "skill_options": {
                "hand_over": {"pre_grasp_distance": 0.025},
                "simulation.place_relative": {
                    "lift_height": 0.010,
                    "cartesian_approach": True,
                },
            },
        },
    },
    tilt_candidates=(0.0, -0.175, 0.175, -0.35, 0.35),
    yaw_candidates=tuple(index * math.pi / 16 for index in range(32)),
)
_HIGH_RELEASE_DEPLOYMENT, _HIGH_RELEASE_ADAPTER = _pen_placement(
    0.166,
    integration_overrides={
        "profile": {
            "skill_options": {
                "simulation.place_relative": {
                    "hand_interp_steps": 12,
                    "release_settle_steps": 15,
                },
            },
        },
    },
)
_PROGRAM = load_task_program(
    _DEPLOYMENT.program_path,
    integration=_DEPLOYMENT.selection,
    validation_context=_ADAPTER.registration.catalog,
)
_HIGH_RELEASE_PROGRAM = replace(
    _PROGRAM,
    integration=_HIGH_RELEASE_DEPLOYMENT.selection,
    program=replace(
        _PROGRAM.program,
        items=tuple(item for item in _PROGRAM.program.items if item.name != "align_pen"),
    ),
)


@register_env(
    "RoboSyn-ItemsHandover-v1",
    max_episode_steps=1500,
    task_program_registration=_ADAPTER.registration,
    task_program_adapter_factory=_ADAPTER,
)
class ItemsHandover(EmbodiedEnv):
    def __init__(self, cfg, **kwargs):
        higher_release = (cfg.extensions or {}).get("higher_pen_release", False)
        if cfg.task_program is None:
            cfg.task_program = _HIGH_RELEASE_PROGRAM if higher_release else _PROGRAM
        if higher_release:
            adapter = kwargs.get("task_program_adapter_factory")
            if adapter is None or adapter is _ADAPTER:
                kwargs["task_program_adapter_factory"] = _HIGH_RELEASE_ADAPTER
        super().__init__(cfg=cfg, **kwargs)
        self._pen = self.sim.get_rigid_object("pen")
        self._holder = self.sim.get_rigid_object("holder")
        self._pen_vertices = self._pen.get_vertices(scale=True)
        self._holder_vertices = self._holder.get_vertices(scale=True)

    def _insertion_metrics(self):
        pen = self._pen.get_local_pose(to_matrix=True)
        holder = self._holder.get_local_pose(to_matrix=True)
        distance = torch.linalg.vector_norm(pen[:, :2, 3] - holder[:, :2, 3], dim=-1)
        pen_z = (self._pen_vertices @ pen[:, :3, :3].transpose(1, 2))[:, :, 2] + pen[
            :, 2, 3, None
        ]
        holder_z = (self._holder_vertices @ holder[:, :3, :3].transpose(1, 2))[
            :, :, 2
        ] + holder[:, 2, 3, None]
        overlap = torch.minimum(
            pen_z.amax(dim=1), holder_z.amax(dim=1)
        ) - torch.maximum(pen_z.amin(dim=1), holder_z.amin(dim=1))
        upright = math.cos(1.309)
        return distance, overlap, (pen[:, 2, 0] > upright) & (holder[:, 2, 1] > upright)

    def compute_task_state(self, **kwargs):
        distance, overlap, upright = self._insertion_metrics()
        keep_running = torch.zeros_like(upright)
        return (
            keep_running,
            keep_running,
            {
                "pen_holder_distance_m": distance,
                "pen_holder_vertical_overlap_m": overlap,
                "pen_and_holder_upright": upright,
            },
        )

    def is_task_success(self, **kwargs):
        distance, overlap, upright = self._insertion_metrics()
        return (distance <= 0.03) & (overlap > 0.08) & upright
