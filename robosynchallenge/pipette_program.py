"""Ground a physically articulated pipette for native pickup, motion and press."""

import math

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_skills import (
    ContactGraspGoal,
    EndEffectorPoseGoal,
    JointPositionGoal,
    JointPositionTarget,
    MoveEndEffector,
    MoveJoints,
    PickUp,
    PressAffordance,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.language.loader import load_task_program
from embodichain.utils.math import axis_angle_to_rotation_matrix

from .task_program_adapter import compose_task_program


def _body_to_grasp(pose):
    # Pinch the upper barrel at the center of its cross-section.
    return pose.new_tensor(
        ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, -0.020), (0, 0, 0, 1))
    )


_PRESS_INCLINATIONS = (55, 65, 70, 75)
_CAP_FACE = 0.00354


def _press_targets(button, left_base, affordance):
    """Generate contact poses with inclined approaches toward the plunger."""
    up = button[..., :3, 0]
    contact = button[..., :3, 3] + _CAP_FACE * up
    radial = contact - left_base[:, None].to(contact)
    tangent = torch.nn.functional.normalize(
        radial - (radial * up).sum(dim=-1, keepdim=True) * up, dim=-1
    )
    tilt = torch.deg2rad(button.new_tensor(_PRESS_INCLINATIONS))
    axes = (
        -up[..., None, :] * tilt.cos()[:, None]
        + tangent[..., None, :] * tilt.sin()[:, None]
    )
    frames = button[..., None, :, :].expand(*axes.shape[:-1], 4, 4).clone()
    frames[..., :3, 2] = axes
    frames[..., :3, 3] = contact[..., None, :]
    poses = affordance.get_press_pose(frames.flatten(0, -3)).reshape_as(frames)
    return poses


def _press_joint_path(robot, context, poses, plunger_axis):
    """Keep the wrist inclined while advancing along the actual joint axis."""
    start = context.robot.qpos[:, robot.get_joint_ids("left_arm")]
    offsets = poses.new_tensor((0.038, 0.010, 0.0, -0.010, -0.025, 0.010, 0.038))
    targets = (
        poses[..., None, :, :].expand(*poses.shape[:-2], len(offsets), 4, 4).clone()
    )
    targets[..., :3, 3] += plunger_axis[..., None, None, :] * offsets[:, None]
    flat = targets.flatten(1, -3)
    success, solution = robot.compute_batch_ik(
        pose=flat,
        joint_seed=start[:, None].expand(-1, flat.shape[1], -1),
        name="left_arm",
        env_ids=context.env_ids,
    )
    success = success.to(torch.bool).reshape(*poses.shape[:-2], len(offsets))
    solution = solution.reshape(*success.shape, start.shape[-1])
    return success.all(dim=-1), solution


class PipetteGrasp(RegisteredSemanticLowerer):
    call_id = "robosyn.grasp_pipette"
    revision = "1"
    target_descriptor = PickUp.descriptor()
    requires_unoccupied_slots = ("primary",)
    preserves_symbolic_state = True

    def create(self, *, simulation, robot, scene_registry, engine):
        return PipetteGrasp()

    def lower(self, call, *, context, bound, option_template):
        pose = context.scene.entities["pipette_body"].pose
        return SemanticLowering(goal=ContactGraspGoal(pose @ _body_to_grasp(pose)))


class ApproachPipette(RegisteredSemanticLowerer):
    call_id = "robosyn.approach_pipette"
    revision = "1"
    target_descriptor = MoveEndEffector.descriptor()
    preserves_symbolic_state = True

    def create(self, *, simulation, robot, scene_registry, engine):
        return ApproachPipette()

    def lower(self, call, *, context, bound, option_template):
        body = context.scene.entities["pipette_body"].pose
        target = body @ _body_to_grasp(body)
        target[:, 2, 3] += 0.09
        return SemanticLowering(goal=EndEffectorPoseGoal(target))


class PositionPipette(RegisteredSemanticLowerer):
    call_id = "robosyn.position_pipette"
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot, beaker_id):
        self.robot = robot
        self.beaker_id = beaker_id
        self.left_base = robot.get_link_pose("left_arm_base")[:, :3].clone()
        self.press_affordance = PressAffordance(press_position=(0.0, 0.0, 0.0))

    def lower(self, call, *, context, bound, option_template):
        body = context.scene.entities["pipette_body"].pose
        beaker = context.scene.entities[self.beaker_id].pose
        motion = bound.binding.resources["primary"].endpoints["motion"]
        arm = motion.runtime_target
        if not isinstance(arm, JointPositionTarget):
            raise TypeError("Pipette transport requires a joint-position arm.")
        arm_ids = self.robot.get_joint_ids("right_arm")
        arm_qpos = context.robot.qpos[:, arm_ids]
        wrist = self.robot.compute_fk(
            qpos=arm_qpos,
            name="right_arm",
            to_matrix=True,
            env_ids=context.env_ids,
        )
        body_to_wrist = torch.linalg.solve(body, wrist)
        upright = body.new_tensor(((0, 0, -1), (0, 1, 0), (1, 0, 0)))
        # The beaker opening admits several upright positions and any yaw.
        yaw = torch.arange(16, device=body.device, dtype=body.dtype) * (math.pi / 8)
        yaw_rotations = axis_angle_to_rotation_matrix(
            yaw[:, None] * body.new_tensor((0.0, 0.0, 1.0))
        )
        axes = body.new_tensor(
            [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        )
        tilt_rotations = axis_angle_to_rotation_matrix(axes * math.radians(15))
        rotations = (yaw_rotations[:, None] @ tilt_rotations @ upright).flatten(0, 1)
        offsets = body.new_tensor(
            [
                (x, y, height)
                for height in (0.15, 0.175, 0.20, 0.225)
                for x, y in ((0, 0), (0.04, 0), (-0.04, 0), (0, 0.04), (0, -0.04))
            ]
        )
        count = len(rotations) * len(offsets)
        targets = body[:, None].expand(-1, count, -1, -1).clone()
        targets[:, :, :3, :3] = rotations.repeat(len(offsets), 1, 1)
        targets[:, :, :3, 3] = beaker[:, None, :3, 3] + offsets.repeat_interleave(
            len(rotations), dim=0
        )
        wrist_targets = targets @ body_to_wrist[:, None]
        seeds = arm_qpos[:, None].expand(-1, count, -1)
        valid, solutions = self.robot.compute_batch_ik(
            pose=wrist_targets,
            joint_seed=seeds,
            name="right_arm",
            env_ids=context.env_ids,
        )
        button = context.scene.entities["pipette_button"].pose
        body_to_button = torch.linalg.solve(body, button)
        button_targets = targets @ body_to_button[:, None]
        press_poses = _press_targets(
            button_targets, self.left_base[context.env_ids], self.press_affordance
        )
        reachable, _ = _press_joint_path(
            self.robot, context, press_poses, button_targets[..., :3, 0]
        )
        receiving_arm_reachable = reachable.any(dim=-1)
        valid = valid.to(torch.bool) & receiving_arm_reachable
        costs = (solutions - seeds).square().sum(dim=-1).masked_fill(~valid, torch.inf)
        rows = torch.arange(context.batch_size, device=body.device)
        chosen = solutions[rows, costs.argmin(dim=1)]
        chosen = torch.where(valid.any(dim=1, keepdim=True), chosen, arm_qpos)
        failed = {
            int(
                context.env_ids[row]
            ): "No upright pipette pose over the beaker is reachable by both arms."
            for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
        }
        closed = chosen.new_zeros(
            (context.batch_size, len(arm.joint_ids) - len(arm_ids))
        )
        target = torch.cat((chosen, closed), dim=-1)
        return SemanticLowering(
            goal=JointPositionGoal(target), failed_preconditions=failed
        )


class PositionPipetteFactory:
    beaker_id = "beaker1"
    call_id = PositionPipette.call_id
    revision = "1"
    target_descriptor = PositionPipette.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return PositionPipette(robot, self.beaker_id)


class PipettePress(RegisteredSemanticLowerer):
    """Lower a vertical plunger stroke to native synchronized joint waypoints."""

    call_id = "robosyn.depress_plunger"
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot):
        self.robot = robot
        self.left_base = robot.get_link_pose("left_arm_base")[:, :3].clone()
        self.affordance = PressAffordance(press_position=(0.0, 0.0, 0.0))

    def lower(self, call, *, context, bound, option_template):
        button = context.scene.entities["pipette_button"].pose
        poses = _press_targets(
            button[:, None], self.left_base[context.env_ids], self.affordance
        )[:, 0]
        valid, solutions = _press_joint_path(
            self.robot, context, poses, button[:, :3, 0]
        )
        rows = torch.arange(context.batch_size, device=button.device)
        stroke = solutions[rows, valid.long().argmax(dim=1)]
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        joint_ids = list(endpoint.runtime_target.joint_ids)
        paths = (
            context.robot.qpos[:, None, joint_ids]
            .expand(-1, stroke.shape[1], -1)
            .clone()
        )
        left = [
            joint_ids.index(index) for index in self.robot.get_joint_ids("left_arm")
        ]
        jaws = [
            joint_ids.index(index)
            for part in ("left_eef", "right_eef")
            for index in self.robot.get_joint_ids(part)
        ]
        paths[..., left] = stroke
        paths[..., jaws] = 0.0
        failures = {
            int(context.env_ids[row]): "The left arm cannot reach the observed plunger."
            for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
        }
        return SemanticLowering(
            goal=JointPositionGoal(paths), failed_preconditions=failures
        )


class PipettePressFactory:
    call_id = PipettePress.call_id
    revision = "1"
    target_descriptor = PipettePress.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return PipettePress(robot)


_DEPLOYMENT, _ADAPTER = compose_task_program(
    "manipulate_pipette",
    (),
    additional_lowerers=(
        ApproachPipette(),
        PipetteGrasp(),
        PositionPipetteFactory(),
        PipettePressFactory(),
    ),
)
_PROGRAM = load_task_program(
    _DEPLOYMENT.program_path,
    integration=_DEPLOYMENT.selection,
    validation_context=_ADAPTER.registration.catalog,
)


@register_env(
    "RoboSyn-ManipulatePipette-v1",
    max_episode_steps=1500,
    task_program_registration=_ADAPTER.registration,
    task_program_adapter_factory=_ADAPTER,
)
class ManipulatePipette(EmbodiedEnv):
    program = _PROGRAM

    def __init__(self, cfg, **kwargs):
        if cfg.task_program is None:
            cfg.task_program = self.program
        super().__init__(cfg=cfg, **kwargs)
        self._pipette = self.sim.get_articulation("pipette")
        self._beaker = self.sim.get_rigid_object("beaker1")
        self._press_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._pressed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        rows = (options or {}).get("reset_ids", slice(None))
        self._press_count[rows] = 0
        self._pressed[rows] = False
        return obs, info

    def _over_upright_beaker(self, beaker=None):
        body = self._pipette.get_local_pose(to_matrix=True)
        beaker = (beaker if beaker is not None else self._beaker).get_local_pose(
            to_matrix=True
        )
        vertical = math.cos(math.radians(20))
        distance = torch.linalg.vector_norm(body[:, :2, 3] - beaker[:, :2, 3], dim=-1)
        return (
            (body[:, 2, 0] > vertical)
            & (beaker[:, 2, 2] > vertical)
            & (distance <= 0.06)
            & (body[:, 2, 3] > beaker[:, 2, 3] + 0.12)
        )

    def compute_task_state(self, **kwargs):
        qpos = self._pipette.get_qpos()[:, 0]
        minimum = self._pipette.get_qpos_limits()[:, 0, 0]
        down = qpos <= 0.66 * minimum + 1.0e-4
        self._press_count += (
            down & ~self._pressed & self._over_upright_beaker()
        ).long()
        self._pressed.copy_(down)
        keep_running = torch.zeros_like(down)
        return keep_running, keep_running, {"pipette_press_count": self._press_count}

    def is_task_success(self, **kwargs):
        return (self._press_count >= 1) & self._over_upright_beaker()
