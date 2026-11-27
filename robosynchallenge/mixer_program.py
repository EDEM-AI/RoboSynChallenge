"""Plan a native switch stroke and measure contact at the mixer."""

import torch

from embodichain.compute.geometry import nearest_half_turn_pose
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_skills import (
    JointPositionGoal,
    MoveJoints,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.language.loader import load_task_program
from embodichain.utils.math import axis_angle_to_rotation_matrix

from .insertion_targets import InsertionPlaceFactory, _closed_jaw_targets
from .task_program_adapter import compose_task_program

_SWITCH_POSITION = (-0.1176, 0.0, 0.056)
_SWITCH_APPROACH = 0.055
_SWITCH_STROKE = 0.019
_SWITCH_CONTACT_ROTATION = (
    (0.34695238, 0.56999016, 0.74480563),
    (0.90711051, -0.00218471, -0.42088687),
    (-0.23827420, 0.82164878, -0.51780206),
)


class MixerPressLowerer(RegisteredSemanticLowerer):
    call_id = "robosyn.press_mixer_switch"
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot):
        self.robot = robot

    def lower(self, call, *, context, bound, option_template):
        target = call.arguments["object"]
        if target != "mixer":
            raise ValueError("Mixer press requires the mixer scene entity.")
        pose = context.scene.entities[target].pose
        motion = bound.binding.resources["primary"].endpoints["motion"].runtime_target
        joints = context.robot.qpos[:, self.robot.get_joint_ids("left_arm")]
        tcp = self.robot.compute_fk(
            qpos=joints, name="left_arm", to_matrix=True, env_ids=context.env_ids
        )
        # The switch point and press axis are fixed. A small roll keeps the
        # Cartesian stroke on one reachable IK branch as the mixer pose varies.
        rolls = pose.new_tensor((0.0, -0.14, 0.14, -0.28, 0.28, -0.42, 0.42))
        local_rotation = pose.new_tensor(_SWITCH_CONTACT_ROTATION) @ (
            axis_angle_to_rotation_matrix(rolls[:, None] * pose.new_tensor((0, 0, 1)))
        )
        contact = pose[:, None].expand(-1, len(rolls), -1, -1).clone()
        contact[..., :3, :3] = pose[:, None, :3, :3] @ local_rotation
        contact[..., :3, 3] = (
            pose[:, :3, 3] + pose[:, :3, :3] @ pose.new_tensor(_SWITCH_POSITION)
        )[:, None]
        contact = nearest_half_turn_pose(
            contact.flatten(0, 1), tcp.repeat_interleave(len(rolls), dim=0)
        )
        stroke = torch.linspace(
            -_SWITCH_APPROACH, _SWITCH_STROKE, 24,
            dtype=pose.dtype, device=pose.device,
        )
        path = contact[:, None].expand(-1, len(stroke), -1, -1).clone()
        path[..., :3, 3] += contact[:, None, :3, 2] * stroke[None, :, None]
        valid, solutions = self.robot.compute_batch_ik(
            pose=path,
            joint_seed=joints.repeat_interleave(len(rolls), dim=0),
            name="left_arm",
            env_ids=context.env_ids.repeat_interleave(len(rolls)),
            continuous=True,
        )
        continuous = solutions.diff(dim=1).abs().amax(dim=(1, 2)) <= 0.20
        feasible = (valid.all(dim=1) & continuous).reshape(context.batch_size, -1)
        score = rolls.abs().expand_as(feasible).masked_fill(~feasible, torch.inf)
        rows = torch.arange(context.batch_size, device=pose.device)
        solved = solutions.reshape(context.batch_size, len(rolls), len(stroke), -1)
        press = solved[rows, score.argmin(dim=1)]
        arm_targets = torch.cat((joints[:, None], press, press[:, :-1].flip(dims=(1,))), dim=1)
        targets = _closed_jaw_targets(
            self.robot, context, list(motion.joint_ids), arm_targets, "left"
        )
        return SemanticLowering(
            goal=JointPositionGoal(targets),
            failed_preconditions={
                int(context.env_ids[row]): "No continuous wrist path reaches the mixer switch."
                for row in (~feasible.any(dim=1)).nonzero().flatten().tolist()
            },
        )


class MixerPressFactory:
    call_id = MixerPressLowerer.call_id
    revision = "1"
    target_descriptor = MixerPressLowerer.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return MixerPressLowerer(robot)


_DEPLOYMENT, _ADAPTER = compose_task_program(
    "mixer_operating",
    (
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
    additional_lowerers=(MixerPressFactory(),),
)
_RANDOM_DEPLOYMENT, _RANDOM_ADAPTER = compose_task_program(
    "mixer_operating",
    (),
    additional_lowerers=(MixerPressFactory(),),
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
_PROGRAM = load_task_program(
    _DEPLOYMENT.program_path,
    integration=_DEPLOYMENT.selection,
    validation_context=_ADAPTER.registration.catalog,
)
_RANDOM_PROGRAM = load_task_program(
    _RANDOM_DEPLOYMENT.program_path,
    integration=_RANDOM_DEPLOYMENT.selection,
    validation_context=_RANDOM_ADAPTER.registration.catalog,
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
            cfg.task_program = _RANDOM_PROGRAM if preserve_beaker_orientation else _PROGRAM
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
