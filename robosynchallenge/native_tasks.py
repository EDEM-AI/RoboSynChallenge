"""EmbodiChain Task Program registrations for the challenge."""

from dataclasses import replace
import math
from pathlib import Path

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_skills import (
    ObjectSemantics,
    Press,
    PressAffordance,
    PressGoal,
    SceneEntityPose,
    sample_initial_articulation_geometry,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.integrations.composition import (
    load_configured_task_program_deployment,
)
from embodichain.lab.task_program.integrations.simulation.environment import (
    SimulationTaskProgramAdapterFactory,
)
from embodichain.utils.utility import load_config



class BellPressLowerer(RegisteredSemanticLowerer):
    call_id = "simulation.press"
    target_descriptor = Press.descriptor()
    requires_unoccupied_slots = ("primary",)
    preserves_symbolic_state = True

    def __init__(self, semantics: ObjectSemantics, arm_base: torch.Tensor):
        self._semantics = semantics
        self._arm_base = arm_base

    def lower(self, call, *, context, bound, option_template):
        target = call.arguments["object"]
        if target != self._semantics.entity_id:
            raise ValueError(
                f"Expected bell press target {self._semantics.entity_id}, got {target}."
            )

        target_pose = context.scene.entities[target].pose[0]
        to_bell = target_pose[:3, 3] - self._arm_base.to(target_pose)
        local_direction = target_pose[:3, :3].T @ to_bell
        surface = self._semantics.affordance
        axis = surface.press_axis.to(target_pose)
        tangent = local_direction - torch.dot(local_direction, axis) * axis
        outward = torch.nn.functional.normalize(tangent, dim=0)
        # Keep the CobotMagic approach angle used by the original expert.
        # Press still derives the approach, contact and retract paths itself.
        tilt = math.radians(20.0)
        contact = PressAffordance(
            press_axis=axis * math.cos(tilt) + outward * math.sin(tilt),
            press_position=surface.press_position,
        )
        semantics = replace(self._semantics, geometry={}, affordance=contact)
        return SemanticLowering(goal=PressGoal(semantics, SceneEntityPose(target)))


class BellPressFactory:
    """Bind the button link geometry to the SDK's native Press skill."""

    call_id = "simulation.press"
    revision = "1"
    target_descriptor = Press.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        bell = simulation.get_articulation("button")
        initial_qpos = torch.zeros_like(bell.get_qpos()[0])
        geometry = sample_initial_articulation_geometry(
            bell,
            "button_cover",
            initial_qpos=initial_qpos,
            initial_qpos_joint_names=bell.joint_names,
            body_scale=bell.cfg.body_scale,
        )
        semantics = ObjectSemantics(
            label="Bell button",
            geometry=geometry.to_object_geometry(),
            entity_id="button_cap",
            affordance=PressAffordance(),
        )
        arm_base = robot.get_link_pose("right_arm_base")
        if len(arm_base) != 1:
            raise ValueError("The Press recovery policy requires one environment.")
        return BellPressLowerer(semantics, arm_base[0, :3].clone())


_PROGRAM_DIR = Path(__file__).resolve().parents[1] / "configs/click_bell/task_program"
_DEPLOYMENT = load_configured_task_program_deployment(
    task_program={
        "program": "program.yaml",
        "integration": {"component": "integration.yaml"},
        "execution_policy": {"component": "execution_policy.yaml"},
    },
    skill_profile=load_config(_PROGRAM_DIR / "skill_profile.yaml"),
    base_dir=_PROGRAM_DIR,
)
_REGISTRATION = replace(
    _DEPLOYMENT.integration.registration,
    registered_semantic_lowerer_factories=(BellPressFactory(),),
)
_ADAPTER = SimulationTaskProgramAdapterFactory(
    _REGISTRATION, planner_config=_DEPLOYMENT.planner_config
)


@register_env(
    "RoboSyn-ClickBell-v1",
    max_episode_steps=361,
    task_program_registration=_REGISTRATION,
    task_program_adapter_factory=_ADAPTER,
)
class ClickBell(EmbodiedEnv):
    """Measure the deepest spring-loaded button travel across the episode."""

    press_depth_threshold = 0.0048

    def __init__(self, cfg, **kwargs):
        if cfg.task_program is None:
            cfg.task_program = _DEPLOYMENT.load_program()
        super().__init__(cfg=cfg, **kwargs)
        self._peak_press_depth = torch.zeros(self.num_envs, device=self.device)
        self._button_joint = self.sim.get_articulation("button").joint_names.index(
            "press"
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        reset_ids = (options or {}).get("reset_ids")
        if reset_ids is None:
            self._peak_press_depth.zero_()
        else:
            self._peak_press_depth[reset_ids] = 0.0
        return obs, info

    def compute_task_state(self, **kwargs):
        qpos = self.sim.get_articulation("button").get_qpos()
        press_depth = -qpos[:, self._button_joint]
        self._peak_press_depth = torch.maximum(self._peak_press_depth, press_depth)
        keep_running = torch.zeros_like(press_depth, dtype=torch.bool)
        return (
            keep_running,
            keep_running,
            {
                "press_depth": press_depth,
                "peak_press_depth": self._peak_press_depth,
                "movement_threshold": self.press_depth_threshold,
            },
        )

    def is_task_success(self, **kwargs):
        return self._peak_press_depth >= self.press_depth_threshold

    def get_demo_episode_metadata(self, env_id):
        metadata = super().get_demo_episode_metadata(env_id)
        metadata["peak_press_depth_m"] = self._peak_press_depth[env_id].item()
        return metadata
