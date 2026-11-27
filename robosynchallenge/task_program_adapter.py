"""Compose SDK Task Programs that need challenge-owned affordance bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.sim.utility.gripper_geometry import (
    build_gripper_collision_geometry,
)
from embodichain.lab.task_program.integrations.composition import (
    load_configured_task_program_deployment,
)
from embodichain.lab.task_program.integrations.invocation_policy import (
    InvocationPolicyBinding,
    task_program_fingerprint,
)
from embodichain.lab.task_program.integrations.simulation.environment import (
    SimulationTaskProgramAdapterFactory,
)
from embodichain.lab.task_program.semantics.calls import (
    RegisteredSemanticCall,
    SemanticCallDescriptor,
)
from embodichain.lab.task_program.semantics.profiles import SkillPolicyPreset
from embodichain.robot_registry import resolve_gripper_defaults
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.utils.utility import load_config

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from embodichain.lab.sim.motion.expansion import CombinedExpansionProfile
    from embodichain.lab.task_program.compiler.program import CompiledTaskProgram
    from embodichain.lab.task_program.integrations.extensions import (
        ParallelCommandSafetyValidatorFactory,
    )
    from embodichain.lab.task_program.runtime.expansion import (
        TaskProgramNominalSourceFactory,
        TaskProgramPlanTransformFactory,
    )

__all__ = ["InvocationMotionPolicyEnv", "compose_task_program"]


class InvocationMotionPolicyEnv(EmbodiedEnv):
    """Select declared motion presets for specific registered semantic calls."""

    call_motion_presets: ClassVar[Mapping[tuple[str, str], str]] = {}

    def create_task_program_bridge(
        self,
        program: CompiledTaskProgram,
        *,
        expansion_profile: CombinedExpansionProfile | None = None,
        expansion_candidate_index: int = 0,
        expansion_recipe_indices: tuple[int, ...] | None = None,
        nominal_source_factory: TaskProgramNominalSourceFactory | None = None,
        plan_transform_factory: TaskProgramPlanTransformFactory | None = None,
    ) -> TaskProgramDemoBridge:
        """Build the standard bridge with any task-owned occurrence selections."""
        occurrences = []
        prerequisites = {}
        for segment in program.iter_segments():
            if segment.contract is not None:
                prerequisites.update(
                    (entry.ref, entry.spec)
                    for entry in segment.contract.prerequisites
                    if entry.spec is not None
                )
            if segment.parallel_block is not None:
                continue
            for compiled_call in segment.calls:
                call = compiled_call.call
                if isinstance(call, RegisteredSemanticCall):
                    preset = self.call_motion_presets.get(
                        (program.program_id, call.call_id)
                    )
                    if preset is not None:
                        invocation_id = (
                            f"{program.program_id}/{segment.segment_id}:"
                            f"{compiled_call.segment_call_index}"
                        )
                        occurrences.append((invocation_id, preset))

        if not occurrences or self.cfg.task_program_execution == "policy":
            return super().create_task_program_bridge(
                program,
                expansion_profile=expansion_profile,
                expansion_candidate_index=expansion_candidate_index,
                expansion_recipe_indices=expansion_recipe_indices,
                nominal_source_factory=nominal_source_factory,
                plan_transform_factory=plan_transform_factory,
            )

        fingerprint = task_program_fingerprint(program)
        bindings = tuple(
            InvocationPolicyBinding(fingerprint, invocation_id, preset)
            for invocation_id, preset in occurrences
        )
        # The SDK's Gym bridge hook does not forward invocation policy bindings.
        bridge = self._checked_task_program_adapter().create_bridge(
            program,
            expansion_profile=expansion_profile,
            candidate_index=expansion_candidate_index,
            recipe_indices=expansion_recipe_indices,
            nominal_source_factory=nominal_source_factory,
            plan_transform_factory=plan_transform_factory,
            isolate_policy_sessions=self.num_envs > 1,
            invocation_policy_bindings=bindings,
        )
        if prerequisites:
            from embodichain.lab.gym.envs.task_program._final_acceptance import (
                _capture_evaluation_frame,
            )
            from embodichain.lab.task_program.integrations.simulation.bindings import (
                _bind_evaluation_joint_names,
            )

            bridge._contract_conditions = _bind_evaluation_joint_names(
                prerequisites, self.robot
            )
            bridge._contract_capture = lambda fields: _capture_evaluation_frame(
                self, fields
            )
            bridge._contract_episode_owner = self.episode_context
        return bridge


def _motion_presets(
    base: SkillPolicyPreset, overrides: Mapping[str, Mapping[str, float]]
) -> tuple[SkillPolicyPreset, ...]:
    return tuple(
        SkillPolicyPreset(
            preset_id=name,
            effect_assurance=base.effect_assurance,
            motion_policy=replace(base.motion_policy, **motion),
            tracking_policy=base.tracking_policy,
            recovery_policy=base.recovery_policy,
            workflow_recovery_policy=base.workflow_recovery_policy,
            runner_cfg=base.runner_cfg,
            effect_monitors=base.effect_monitors,
            required_planner=base.required_planner,
            skill_option_templates=base.skill_option_templates,
        )
        for name, motion in overrides.items()
    )


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
    execution_policy_overrides=None,
    motion_presets: Mapping[str, Mapping[str, float]] | None = None,
    parallel_safety_factory: ParallelCommandSafetyValidatorFactory | None = None,
):
    folder = Path(__file__).resolve().parents[1] / "configs" / task
    evaluation = folder / "task_program/evaluation.yaml"
    skill_profile = load_config(folder / "clear/cobotmagic.yaml")["skill_profile"]
    component = (
        "integration.yaml"
        if integration_component is None
        else {
            "inline": integration_component,
        }
    )
    integration = {"component": component}
    if integration_overrides is not None:
        integration["overrides"] = integration_overrides
    execution_policy = {"component": "execution_policy.yaml"}
    if execution_policy_overrides is not None:
        execution_policy["overrides"] = execution_policy_overrides
    deployment = load_configured_task_program_deployment(
        task_program={
            "program": "program.yaml",
            "integration": integration,
            "execution_policy": execution_policy,
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
    profile = configured.robot_profile_binding
    if motion_presets:
        base = next(
            preset
            for preset in profile.presets
            if preset.preset_id == deployment.selection.runtime_preset
        )
        profile = replace(
            profile, presets=profile.presets + _motion_presets(base, motion_presets)
        )
    registration = replace(
        configured,
        call_catalog=catalog,
        robot_profile_binding=profile,
        parallel_safety_factory=parallel_safety_factory
        or configured.parallel_safety_factory,
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
