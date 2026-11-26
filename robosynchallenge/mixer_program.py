"""Bind the mixer switch to native Press and measure contact at the switch."""

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_skills import (
    ObjectSemantics,
    Press,
    PressAffordance,
    PressGoal,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)

from .insertion_targets import InsertionPlaceFactory
from .task_program_adapter import compose_task_program

_SWITCH_POSITION = (-0.1176, 0.0, 0.056)
_SWITCH_CONTACT_ROTATION = (
    (0.34695238, 0.56999016, 0.74480563),
    (0.90711051, -0.00218471, -0.42088687),
    (-0.23827420, 0.82164878, -0.51780206),
)


class MixerSwitchAffordance(PressAffordance):
    """Keep the calibrated wrist orientation fixed to the mixer body."""

    def get_press_pose(self, target_pose, press_position=None):
        contact = super().get_press_pose(target_pose, press_position)
        contact[:, :3, :3] = target_pose[:, :3, :3] @ contact.new_tensor(
            _SWITCH_CONTACT_ROTATION
        )
        return contact


class MixerPressLowerer(RegisteredSemanticLowerer):
    call_id = "simulation.press"
    target_descriptor = Press.descriptor()
    requires_unoccupied_slots = ("primary",)
    preserves_symbolic_state = True

    def lower(self, call, *, context, bound, option_template):
        target = call.arguments["object"]
        if target != "mixer":
            raise ValueError("Mixer press requires the mixer scene entity.")
        pose = context.scene.entities[target].pose
        semantics = ObjectSemantics(
            affordance=MixerSwitchAffordance(
                press_axis=pose.new_tensor(_SWITCH_CONTACT_ROTATION)[:, 2],
                press_position=_SWITCH_POSITION,
            ),
            geometry={},
            entity_id=target,
        )
        # The mixer moves under contact. Complete this stroke from its initial pose.
        return SemanticLowering(goal=PressGoal(semantics, pose.clone()))


class MixerPressFactory:
    call_id = "simulation.press"
    revision = "1"
    target_descriptor = Press.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        return MixerPressLowerer()


_DEPLOYMENT, _ADAPTER = compose_task_program(
    "mixer_operating",
    (
        MixerPressFactory(),
        InsertionPlaceFactory(
            "beaker",
            "mixer",
            (0, 0, 0.09),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            arm_side="right",
            above_opening=0.020,
            offset_in_world=True,
        ),
    ),
)
_RANDOM_DEPLOYMENT, _RANDOM_ADAPTER = compose_task_program(
    "mixer_operating",
    (MixerPressFactory(),),
    integration_overrides={
        "profile": {
            "skill_options": {
                "pick": {"grasp_settle_steps": 15, "cartesian_lift": False},
                "simulation.place_relative": {
                    "lift_height": 0.08,
                    "preserve_current_object_orientation": True,
                },
            },
        },
    },
)


@register_env(
    "RoboSyn-MixerOperating-v1",
    max_episode_steps=1500,
    task_program_registration=_ADAPTER.registration,
    task_program_adapter_factory=_ADAPTER,
)
class MixerOperating(EmbodiedEnv):
    def __init__(self, cfg, **kwargs):
        preserve_beaker_orientation = (cfg.extensions or {}).get(
            "preserve_beaker_orientation", False
        )
        if cfg.task_program is None:
            deployment = _RANDOM_DEPLOYMENT if preserve_beaker_orientation else _DEPLOYMENT
            cfg.task_program = deployment.load_program()
        if preserve_beaker_orientation:
            adapter = kwargs.get("task_program_adapter_factory")
            if adapter is None or adapter is _ADAPTER:
                kwargs["task_program_adapter_factory"] = _RANDOM_ADAPTER
        super().__init__(cfg=cfg, **kwargs)
        self._pressed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contact_sensor = self.sim.get_sensor("beaker_mixer_button_contact")
        self._mixer_ids = self._contact_sensor.get_actor_ids("beaker_mixer")
        self._finger_ids = self._contact_sensor.get_actor_ids(
            self.robot.uid, ("left_link7", "left_link8")
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        ids = (options or {}).get("reset_ids")
        if ids is None:
            self._pressed.zero_()
        else:
            self._pressed[ids] = False
        return obs, info

    def _placement(self):
        mixer = self.sim.get_rigid_object("beaker_mixer").get_local_pose(to_matrix=True)
        beaker = self.sim.get_rigid_object("beaker").get_local_pose(to_matrix=True)
        distance = torch.linalg.vector_norm(beaker[:, :2, 3] - mixer[:, :2, 3], dim=-1)
        upright = beaker[:, 2, 2] > 0.5
        return mixer, distance, upright

    def compute_task_state(self, **kwargs):
        mixer, distance, upright = self._placement()
        contact = self._contact_sensor.get_data()
        point = mixer[:, :3, :3] @ mixer.new_tensor(_SWITCH_POSITION) + mixer[:, :3, 3]
        a, b = contact["user_ids"].unbind(dim=-1)
        a_is_finger = (a[:, :, None] == self._finger_ids[:, None, :]).any(dim=-1)
        b_is_finger = (b[:, :, None] == self._finger_ids[:, None, :]).any(dim=-1)
        actors = ((a == self._mixer_ids) & b_is_finger) | (
            (b == self._mixer_ids) & a_is_finger
        )
        near_switch = (
            torch.linalg.vector_norm(contact["position"] - point[:, None], dim=-1)
            <= 0.01
        )
        self._pressed |= (
            contact["is_valid"]
            & contact["impulse_valid"]
            & actors
            & near_switch
            & (contact["impulse"] >= 0.01)
        ).any(dim=1)
        keep_running = torch.zeros_like(self._pressed)
        return (
            keep_running,
            keep_running,
            {
                "beaker_mixer_distance_m": distance,
                "beaker_upright": upright,
                "switch_contact": self._pressed,
            },
        )

    def is_task_success(self, **kwargs):
        _, distance, upright = self._placement()
        return upright & (distance <= 0.08) & self._pressed
