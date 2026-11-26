"""Evaluate the same physical placement predicates for experts and policies."""

from dataclasses import replace
from pathlib import Path

import torch

from embodichain.lab.evaluation import EvaluationStatus
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_skills import GraspGoal, PickUp
from embodichain.lab.task_program.compiler.lowering import (
    HeldObjectRelation,
    RegisteredHeldObjectEffect,
    RegisteredSemanticEffect,
    RegisteredSemanticLowerer,
    SemanticEffectKind,
    SemanticLowering,
)
from embodichain.lab.task_program.language.loader import load_task_program
from embodichain.lab.task_program.semantics import SceneObjectRef
from embodichain.utils.utility import load_config

from .insertion_targets import (
    InsertionAlignmentFactory,
    InsertionPlaceFactory,
    InsertionLoweringFactory,
)
from .task_program_adapter import compose_task_program


class MeasuredPlacementEnv(EmbodiedEnv):
    deployment = None
    adapter = None
    simulation_uids = {}
    program = None

    def __init__(self, cfg, **kwargs):
        if cfg.task_program is None:
            cfg.task_program = self.program or self.deployment.load_program()
        super().__init__(cfg=cfg, **kwargs)
        registration = self.adapter.registration
        self._placement_rules = {
            ref: registration.evaluation_conditions[ref]
            for ref in registration.evaluation_final_refs
        }
        fields = {
            field
            for rule in self._placement_rules.values()
            for field in rule.required_fields
        }
        self._placement_fields = {}
        for field in fields:
            namespace, entity, kind = field.split("/")
            if namespace != "rigid_objects" or kind not in {"pose", "frame"}:
                raise ValueError(f"Unsupported placement measurement field {field!r}.")
            uid = self.simulation_uids.get(entity, entity)
            self._placement_fields[field] = (self.sim.get_rigid_object(uid), kind)

    def is_task_success(self, **kwargs):
        values = {
            field: (
                asset.get_local_pose()
                if kind == "frame"
                else asset.get_local_pose()[:, :3]
            )
            for field, (asset, kind) in self._placement_fields.items()
        }
        satisfied = [
            rule.evaluate(values, batch_size=self.num_envs, device=self.device)[
                ref
            ].status
            == EvaluationStatus.SATISFIED
            for ref, rule in self._placement_rules.items()
        ]
        return torch.stack(satisfied).all(dim=0)


class CutleryPickup(RegisteredSemanticLowerer):
    """Ground object-local fork and spoon grasps for the native PickUp skill."""

    call_id = "robosyn.pick_cutlery"
    target_descriptor = PickUp.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH
    requires_unoccupied_slots = ("primary",)
    grasp_frames = {
        "fork": (
            (0.98993, -0.00666, 0.14143, 0.0026),
            (0.00864, -0.99419, -0.10732, -0.0025),
            (0.14133, 0.10746, -0.98411, -0.0016),
            (0, 0, 0, 1),
        ),
        "spoon": (
            (0.95458, 0.01288, 0.29769, -0.0109),
            (0.01664, -0.99981, -0.01010, -0.0018),
            (0.29751, 0.01459, -0.95461, -0.0018),
            (0, 0, 0, 1),
        ),
    }

    def __init__(self, semantics):
        self.semantics = semantics

    def lower(self, call, *, context, bound, option_template):
        object_id = call.arguments["object"]
        pose = context.scene.entities[object_id].pose
        grasp = pose @ pose.new_tensor(self.grasp_frames[object_id])
        return SemanticLowering(
            goal=GraspGoal(semantics=self.semantics[object_id], grasp_xpos=grasp),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=object_id,
                        slot_id="primary",
                    ),
                ),
            ),
        )


class CutleryPickupFactory:
    call_id = CutleryPickup.call_id
    revision = "1"
    target_descriptor = CutleryPickup.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        semantics = {}
        for object_id in CutleryPickup.grasp_frames:
            entity = scene_registry.resolve(object_id, expected_type=SceneObjectRef)
            affordance = scene_registry.resolve_affordance(
                entity, capability="affordance.grasp", explicit=f"{object_id}_grasp"
            )
            semantics[object_id] = scene_registry.object_semantics(
                entity, affordance=affordance
            )
        return CutleryPickup(semantics)


_TABLE_DEPLOYMENT, _TABLE_ADAPTER = compose_task_program(
    "table_rearrangement", (), additional_lowerers=(CutleryPickupFactory(),)
)
_TABLE_PROGRAM = load_task_program(
    _TABLE_DEPLOYMENT.program_path,
    integration=_TABLE_DEPLOYMENT.selection,
    validation_context=_TABLE_ADAPTER.registration.catalog,
)
_TUBE_PLACE = InsertionPlaceFactory(
    "test_tube",
    "rack",
    (0.01105, -0.0001, 0.115),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    follow_receptacle_vertical=True,
    above_opening=0.055,
)


def _tube_deployment(integration_component=None):
    deployment, adapter = compose_task_program(
        "sample_loading",
        (replace(_TUBE_PLACE, release_at_current_pose=True),),
        additional_lowerers=(
            InsertionAlignmentFactory(_TUBE_PLACE),
            InsertionLoweringFactory(_TUBE_PLACE),
        ),
        use_scene_grasp_options=True,
        integration_component=integration_component,
    )
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=adapter.registration.catalog,
    )
    return deployment, adapter, program


_TUBE_DEPLOYMENT, _TUBE_ADAPTER, _TUBE_PROGRAM = _tube_deployment()
_SHALLOW_TUBE_INTEGRATION = load_config(
    Path(__file__).resolve().parents[1] / "configs/sample_loading/task_program/integration.yaml"
)
_SHALLOW_TUBE_INTEGRATION["profile"]["skill_options"]["hand_over"]["pre_grasp_distance"] = 0.025
_SHALLOW_TUBE_PRESENTATION = next(
    route
    for route in _SHALLOW_TUBE_INTEGRATION["runtime_services"]["registered_semantic_lowerers"]
    if route.get("target_id") == "handover_presentation"
)
_SHALLOW_TUBE_PRESENTATION["relative_pose"]["quaternion_xyzw"] = [
    0.0,
    0.25881904510252074,
    0.0,
    0.9659258262890683,
]
_, _SHALLOW_TUBE_ADAPTER, _SHALLOW_TUBE_PROGRAM = _tube_deployment(_SHALLOW_TUBE_INTEGRATION)


@register_env(
    "RoboSyn-TableRearrangement-v1",
    max_episode_steps=1400,
    task_program_registration=_TABLE_ADAPTER.registration,
    task_program_adapter_factory=_TABLE_ADAPTER,
)
class TableRearrangement(MeasuredPlacementEnv):
    deployment = _TABLE_DEPLOYMENT
    adapter = _TABLE_ADAPTER
    program = _TABLE_PROGRAM


@register_env(
    "RoboSyn-SampleLoading-v1",
    max_episode_steps=1500,
    task_program_registration=_TUBE_ADAPTER.registration,
    task_program_adapter_factory=_TUBE_ADAPTER,
)
class SampleLoading(MeasuredPlacementEnv):
    deployment = _TUBE_DEPLOYMENT
    adapter = _TUBE_ADAPTER
    program = _TUBE_PROGRAM
    simulation_uids = {"test_tube": "cube"}

    def __init__(self, cfg, **kwargs):
        if (cfg.extensions or {}).get("shallow_tube_handover", False):
            if cfg.task_program is None:
                cfg.task_program = _SHALLOW_TUBE_PROGRAM
            adapter = kwargs.get("task_program_adapter_factory")
            if adapter is None or adapter is _TUBE_ADAPTER:
                kwargs["task_program_adapter_factory"] = _SHALLOW_TUBE_ADAPTER
        super().__init__(cfg=cfg, **kwargs)
        self._tube = self.sim.get_rigid_object("cube")
        self._rack = self.sim.get_rigid_object("rack")

    def compute_task_state(self, **kwargs):
        tube = self._tube.get_local_pose(to_matrix=True)
        rack = self._rack.get_local_pose(to_matrix=True)
        displacement = tube[:, :3, 3] - rack[:, :3, 3]
        keep_running = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return (
            keep_running,
            keep_running,
            {
                "tube_rack_planar_distance_m": displacement[:, :2].norm(dim=1),
                "tube_rack_height_difference_m": displacement[:, 2],
                "tube_tilt_rad": tube[:, 2, 2].clamp(-1, 1).acos(),
                "rack_tilt_rad": rack[:, 2, 2].clamp(-1, 1).acos(),
            },
        )
