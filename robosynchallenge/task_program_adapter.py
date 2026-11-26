"""Compose SDK Task Programs that need challenge-owned affordance bindings."""

from dataclasses import replace
from pathlib import Path
from functools import partial

from embodichain.lab.sim.utility.gripper_geometry import (
    build_gripper_collision_geometry,
)
from embodichain.lab.task_program.integrations.composition import (
    load_configured_task_program_deployment,
)
from embodichain.lab.task_program.integrations.simulation.environment import (
    SimulationTaskProgramAdapterFactory,
)
from embodichain.lab.task_program.semantics.calls import (
    RegisteredSemanticCall,
    SemanticCallDescriptor,
)
from embodichain.robot_registry import resolve_gripper_defaults
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.utils.utility import load_config


def _cobotmagic_grasp_generator(
    *, point_spacing=0.012, filter_ground_collision=True, **algorithm
):
    options = {
        "sample_count": 1500,
        "max_candidates": 48,
        "approach_direction_samples": 4,
    }
    options.update(algorithm)
    gripper = resolve_gripper_defaults("RoboSynCobotMagicV100Gripper")
    return AntipodalGraspPoseGenerator(
        model_id=gripper["model_id"],
        min_opening_width=gripper["min_opening_width"],
        max_opening_width=gripper["max_opening_width"],
        collision_geometry=build_gripper_collision_geometry(
            gripper, point_spacing=point_spacing
        ),
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(**options),
        collision_cfg=ParallelJawGraspCollisionCfg(
            filter_ground_collision=filter_ground_collision
        ),
    )


def compose_task_program(
    task,
    lowerer_factories,
    *,
    additional_lowerers=(),
    use_configured_adapter=False,
    use_scene_grasp_options=False,
    integration_overrides=None,
    integration_component=None,
):
    folder = Path(__file__).resolve().parents[1] / "configs" / task
    evaluation = folder / "task_program/evaluation.yaml"
    skill_profile = load_config(folder / "clear/cobotmagic.yaml")["skill_profile"]
    component = "integration.yaml" if integration_component is None else {
        "inline": integration_component,
    }
    integration = {"component": component}
    if integration_overrides is not None:
        integration["overrides"] = integration_overrides
    deployment = load_configured_task_program_deployment(
        task_program={
            "program": "program.yaml",
            "integration": integration,
            "execution_policy": {"component": "execution_policy.yaml"},
        },
        skill_profile=skill_profile,
        base_dir=folder / "task_program",
        evaluation=load_config(evaluation) if evaluation.is_file() else None,
    )
    if use_configured_adapter:
        return deployment, deployment.integration.adapter_factory
    configured = deployment.integration.registration
    replacements = {factory.call_id: factory for factory in lowerer_factories}
    factories = tuple(
        replacements.pop(factory.call_id, factory)
        for factory in configured.registered_semantic_lowerer_factories
    )
    if replacements:
        raise ValueError(
            f"Undeclared semantic calls for {task}: {sorted(replacements)}"
        )
    catalog = configured.call_catalog
    for factory in additional_lowerers:
        catalog = catalog.with_descriptor(
            SemanticCallDescriptor(
                factory.call_id, RegisteredSemanticCall, factory.target_descriptor
            )
        )
    registration = replace(
        configured,
        call_catalog=catalog,
        registered_semantic_lowerer_factories=factories + tuple(additional_lowerers),
    )
    grasp_factories = {
        part: _cobotmagic_grasp_generator for part in ("left_eef", "right_eef")
    }
    if use_scene_grasp_options:
        declared = skill_profile["runtime_services"]["grasp_pose_generators"]
        integration_data = (
            load_config(folder / "task_program/integration.yaml")
            if integration_component is None
            else integration_component
        )
        overrides = integration_data["runtime_services"].get(
            "grasp_pose_generator_overrides", {}
        )
        for part, factory in grasp_factories.items():
            options = {
                key: value
                for key, value in declared[part].items()
                if key not in {"kind", "gripper"}
            }
            options.update(overrides.get(part, {}))
            grasp_factories[part] = partial(factory, **options)
    return deployment, SimulationTaskProgramAdapterFactory(
        registration,
        grasp_pose_generator_factories=grasp_factories,
        planner_config=deployment.planner_config,
    )
