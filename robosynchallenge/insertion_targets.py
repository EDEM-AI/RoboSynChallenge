"""Ground native Place targets at the open end of the challenge's containers."""

from dataclasses import dataclass
import math

import torch

from embodichain.lab.sim.atomic_skills import (
    JointPositionGoal,
    MoveJoints,
    Place,
    PlaceGoal,
)
from embodichain.lab.task_program.compiler.lowering import (
    HeldObjectRelation,
    RegisteredHeldObjectEffect,
    RegisteredSemanticEffect,
    RegisteredSemanticLowerer,
    SemanticEffectKind,
    SemanticLowering,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix


def cylinder_grasp_frame(long_axis):
    """Keep the approach above a cylinder regardless of its roll on the table."""
    down = long_axis.new_tensor((0.0, 0.0, -1.0))
    side = long_axis.new_tensor((0.0, -1.0, 0.0))
    direction = torch.where(long_axis[:, 2:].abs() < 0.95, down, side)
    along = (long_axis * direction).sum(dim=-1, keepdim=True)
    approach = torch.nn.functional.normalize(direction - along * long_axis, dim=-1)
    closing = torch.linalg.cross(approach, long_axis)
    return torch.stack((long_axis, closing, approach), dim=-1)


def _shaft_in_gripper(object_to_tcp):
    offset = object_to_tcp[:, :3, 3]
    return (offset[:, 0].abs() < 0.065) & (
        torch.linalg.vector_norm(offset[:, 1:], dim=-1) < 0.025
    )


@dataclass(frozen=True)
class InsertionPlaceFactory:
    object_id: str
    receptacle_id: str
    release_offset: tuple[float, float, float]
    upright_rotation: tuple[tuple[float, float, float], ...]
    above_opening: float = 0.055
    arm_side: str = "left"
    follow_receptacle_vertical: bool = False
    release_at_current_pose: bool = False
    preserve_object_rotation: bool = False
    offset_in_world: bool = False
    yaw_candidates: tuple[float, ...] = tuple(
        index * math.pi / 8 for index in range(16)
    )
    tilt_candidates: tuple[float, ...] = (0.0,)
    insertion_axis: tuple[float, float, float] | None = None
    opening_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    call_id = "simulation.place_relative"
    revision = "1"
    target_descriptor = Place.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        return InsertionPlace(robot, self)

    def opening_position(self, receptacle):
        offset = receptacle.new_tensor(self.opening_offset)
        return receptacle[:, :3, 3] + receptacle[:, :3, :3] @ offset

    def lift_targets(self, poses, height):
        lifted = poses.clone()
        if self.insertion_axis is None:
            lifted[..., 2, 3] += height
        else:
            axis = poses[..., :3, :3] @ poses.new_tensor(self.insertion_axis)
            lifted[..., :3, 3] += height * axis / axis[..., 2:].clamp_min(0.1)
        return lifted

    def release_targets(self, receptacle, object_pose):
        # Rotating around vertical leaves the insertion axis and drop point fixed.
        yaw = object_pose.new_tensor(self.yaw_candidates).repeat_interleave(
            len(self.tilt_candidates)
        )
        pitch = object_pose.new_tensor(self.tilt_candidates).repeat(
            len(self.yaw_candidates)
        )
        yaw_rotation = axis_angle_to_rotation_matrix(
            yaw[:, None] * object_pose.new_tensor((0.0, 0.0, 1.0))
        )
        orientation = (
            object_pose[:, None, :3, :3]
            if self.preserve_object_rotation
            else object_pose.new_tensor(self.upright_rotation)
        )
        pitch_rotation = axis_angle_to_rotation_matrix(
            pitch[:, None] * object_pose.new_tensor((0.0, 1.0, 0.0))
        )
        rotation = yaw_rotation @ pitch_rotation @ orientation
        targets = receptacle[:, None].expand(-1, len(yaw), -1, -1).clone()
        if self.follow_receptacle_vertical:
            rotation = receptacle[:, None, :3, :3] @ rotation
        targets[..., :3, :3] = rotation
        offset = object_pose.new_tensor(self.release_offset)
        if not self.offset_in_world:
            offset = receptacle[:, :3, :3] @ offset
        targets[..., :3, 3] = (receptacle[:, :3, 3] + offset)[:, None]
        if self.insertion_axis is not None:
            # Aim the shaft through the center of the actual opening even if
            # the wrist needs a slightly tilted placement.
            axis = rotation @ object_pose.new_tensor(self.insertion_axis)
            opening = self.opening_position(receptacle)
            height = targets[..., 2, 3] - opening[:, None, 2]
            targets[..., :2, 3] = opening[:, None, :2] + axis[..., :2] * (
                height / axis[..., 2].clamp_min(0.1)
            )[..., None]
        return targets, pitch


class InsertionPlace(RegisteredSemanticLowerer):
    call_id = InsertionPlaceFactory.call_id
    target_descriptor = InsertionPlaceFactory.target_descriptor
    effect_contract_kind = SemanticEffectKind.RELEASE

    def __init__(self, robot, placement):
        self.robot = robot
        self.placement = placement

    def select_wrist_path(self, context, control_part, lift_height, *, preorient=False):
        placement = self.placement
        joints = context.robot.qpos[:, self.robot.get_joint_ids(control_part)]
        tcp = self.robot.compute_fk(
            qpos=joints, name=control_part, to_matrix=True, env_ids=context.env_ids
        )
        object_pose = context.scene.entities[placement.object_id].pose
        receptacle = context.scene.entities[placement.receptacle_id].pose
        targets, pitch = placement.release_targets(receptacle, object_pose)
        object_to_tcp = torch.linalg.solve(object_pose, tcp)
        release_tcp = targets @ object_to_tcp[:, None]
        above_tcp = placement.lift_targets(targets, placement.above_opening) @ object_to_tcp[:, None]
        clearance_tcp = (
            placement.lift_targets(targets, placement.above_opening + lift_height)
            @ object_to_tcp[:, None]
        )
        retract_tcp = placement.lift_targets(targets, lift_height) @ object_to_tcp[:, None]
        waypoints = [clearance_tcp, above_tcp, release_tcp, retract_tcp]
        if preorient:
            # Turn the held pen while it is still in the open workspace.
            # The translated approach then keeps the tip above the holder.
            turn = targets.clone()
            anchor = object_pose[:, :3, 3] + tcp.new_tensor((-0.018, 0.020, 0.0))
            turn[..., :3, 3] = anchor[:, None]
            waypoints.insert(0, turn @ object_to_tcp[:, None])
        waypoint_count = len(waypoints)
        poses = torch.stack(waypoints, dim=2).flatten(1, 2)
        seeds = joints[:, None].expand(-1, poses.shape[1], -1)
        valid, solutions = self.robot.compute_batch_ik(
            pose=poses,
            joint_seed=seeds,
            name=control_part,
            env_ids=context.env_ids,
        )
        valid = valid.reshape(context.batch_size, len(pitch), waypoint_count).bool()
        reachable = valid.all(dim=-1)
        if preorient:
            use_turn = reachable.any(dim=1, keepdim=True)
            reachable = torch.where(use_turn, reachable, valid[..., 1:].all(dim=-1))
            reachable &= _shaft_in_gripper(object_to_tcp)[:, None]
        costs = (
            (solutions - seeds)
            .square()
            .sum(dim=-1)
            .reshape(context.batch_size, len(pitch), waypoint_count)
            .masked_fill(~valid, 0)
            .sum(dim=-1)
            .masked_fill(~reachable, torch.inf)
        )
        upright = pitch == 0
        upright_available = (reachable & upright).any(dim=1, keepdim=True)
        costs = costs.masked_fill(upright_available & ~upright, torch.inf)
        rows = torch.arange(context.batch_size, device=tcp.device)
        chosen = costs.argmin(dim=-1)
        release = torch.stack(
            (above_tcp[rows, chosen], release_tcp[rows, chosen]), dim=1
        )
        failures = {
            int(
                context.env_ids[row]
            ): "No reachable upright release pose at the opening."
            for row in (~reachable.any(dim=1)).nonzero().flatten().tolist()
        }
        joint_path = solutions.reshape(context.batch_size, len(pitch), waypoint_count, -1)[
            rows, chosen
        ]
        if preorient:
            joint_path[:, 0] = torch.where(
                valid[rows, chosen, 0:1], joint_path[:, 0], joints
            )
        return release, joint_path, failures

    def lower(self, call, *, context, bound, option_template):
        arm = bound.binding.resources["primary"].endpoints["motion"].runtime_target
        placement = self.placement
        if placement.release_at_current_pose:
            release = self.robot.compute_fk(
                qpos=context.robot.qpos[:, list(arm.joint_ids)],
                name=arm.control_part,
                to_matrix=True,
                env_ids=context.env_ids,
            )
            failures = {}
        else:
            release, _, failures = self.select_wrist_path(
                context, arm.control_part, option_template.lift_height
            )
        return SemanticLowering(
            goal=PlaceGoal(release),
            failed_preconditions=failures,
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.RELEASE,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.DETACHED,
                        object_id=placement.object_id,
                        slot_id="primary",
                        release_before_segment="retract",
                        retain_during_segments=("approach",),
                    ),
                ),
            ),
        )


@dataclass(frozen=True)
class InsertionAlignmentFactory:
    placement: InsertionPlaceFactory
    lift_height: float = 0.025
    include_release_pose: bool = False
    preorient: bool = False

    call_id = "robosyn.align_for_insertion"
    revision = "1"
    target_descriptor = MoveJoints.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        return InsertionAlignment(
            InsertionPlace(robot, self.placement),
            self.lift_height,
            self.include_release_pose,
            self.preorient,
        )


class InsertionAlignment(RegisteredSemanticLowerer):
    call_id = InsertionAlignmentFactory.call_id
    target_descriptor = InsertionAlignmentFactory.target_descriptor
    preserves_symbolic_state = True

    def __init__(self, placement, lift_height, include_release_pose, preorient):
        self.placement = placement
        self.lift_height = lift_height
        self.include_release_pose = include_release_pose
        self.preorient = preorient

    def lower(self, call, *, context, bound, option_template):
        robot = self.placement.robot
        motion = bound.binding.resources["primary"].endpoints["motion"]
        joint_ids = list(motion.runtime_target.joint_ids)
        side = self.placement.placement.arm_side
        _, joints, failures = self.placement.select_wrist_path(
            context, f"{side}_arm", self.lift_height, preorient=self.preorient
        )
        waypoint_count = 3 if self.include_release_pose else 2
        waypoint_count += self.preorient
        path = _closed_jaw_targets(
            robot, context, joint_ids, joints[:, :waypoint_count], side
        )
        return SemanticLowering(
            goal=JointPositionGoal(path), failed_preconditions=failures
        )


@dataclass(frozen=True)
class InsertionLoweringFactory:
    placement: InsertionPlaceFactory

    call_id = "robosyn.lower_into_receptacle"
    revision = "1"
    target_descriptor = MoveJoints.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        return InsertionLowering(InsertionPlace(robot, self.placement))


class InsertionLowering(RegisteredSemanticLowerer):
    call_id = InsertionLoweringFactory.call_id
    target_descriptor = InsertionLoweringFactory.target_descriptor
    preserves_symbolic_state = True

    def __init__(self, placement):
        self.placement = placement

    def lower(self, call, *, context, bound, option_template):
        robot = self.placement.robot
        motion = bound.binding.resources["primary"].endpoints["motion"]
        joint_ids = list(motion.runtime_target.joint_ids)
        placement = self.placement.placement
        side = placement.arm_side
        control_part = f"{side}_arm"
        arm_ids = [
            joint_ids.index(index) for index in robot.get_joint_ids(control_part)
        ]
        receptacle = context.scene.entities[placement.receptacle_id].pose
        held_object = context.scene.entities[placement.object_id].pose
        current_joints = context.robot.qpos[:, joint_ids]
        arm_joints = current_joints[:, arm_ids]
        tcp = robot.compute_fk(
            qpos=arm_joints,
            name=control_part,
            to_matrix=True,
            env_ids=context.env_ids,
        )
        release_offset = tcp.new_tensor(placement.release_offset)
        destination = receptacle[:, :3, 3] + receptacle[:, :3, :3] @ release_offset
        object_targets = held_object[:, None].expand(-1, 3, -1, -1).clone()
        heights = tcp.new_tensor(
            (placement.above_opening, placement.above_opening / 2, 0)
        )
        if placement.insertion_axis is None:
            current_axis = held_object[:, :3, 2]
            rack_axis = receptacle[:, :3, 2]
            axis = torch.linalg.cross(current_axis, rack_axis)
            norm = axis.norm(dim=1)
            angle = torch.atan2(norm, (current_axis * rack_axis).sum(dim=1))
            correction = axis * (angle / norm.clamp_min(1e-6))[:, None]
            rotations = axis_angle_to_rotation_matrix(correction)
            object_targets[..., :3, :3] = (rotations @ held_object[:, :3, :3])[:, None]
            object_targets[..., :3, 3] = (
                destination[:, None] + heights[None, :, None] * rack_axis[:, None]
            )
            aligned = torch.ones_like(arm_joints[:, 0], dtype=torch.bool)
        else:
            # The pen may need a small tilt for wrist clearance. Translate
            # along the measured shaft so it stays centered throughout insertion.
            shaft = held_object[:, :3, :3] @ tcp.new_tensor(placement.insertion_axis)
            opening = placement.opening_position(receptacle)
            height = destination[:, 2, None] + 0.6 * heights[None] - opening[:, 2, None]
            travel = height / shaft[:, 2, None].clamp_min(0.1)
            object_targets[..., :3, 3] = opening[:, None] + travel[..., None] * shaft[:, None]
            aligned = shaft[:, 2] > 0.75
        object_to_tcp = torch.linalg.solve(held_object, tcp)
        wrist_targets = object_targets @ object_to_tcp[:, None]
        valid, wrist = robot.compute_batch_ik(
            pose=wrist_targets,
            joint_seed=arm_joints[:, None].expand(-1, 3, -1),
            name=control_part,
            env_ids=context.env_ids,
        )
        ready = valid.all(dim=1) & aligned
        if placement.insertion_axis is not None and not ready.all():
            _, alternate, unreachable = self.placement.select_wrist_path(
                context, control_part, lift_height=0.025
            )
            wrist = torch.where(ready[:, None, None], wrist, alternate[:, :3])
            alternate_ready = torch.tensor(
                [int(env_id) not in unreachable for env_id in context.env_ids],
                device=ready.device,
            )
            ready |= alternate_ready
        if placement.insertion_axis is not None:
            ready &= _shaft_in_gripper(object_to_tcp)
        failures = {
            int(
                context.env_ids[row]
            ): "The aligned insertion path is outside the arm's reach."
            for row in (~ready).nonzero().flatten().tolist()
        }
        path = _closed_jaw_targets(robot, context, joint_ids, wrist, side)
        return SemanticLowering(
            goal=JointPositionGoal(path), failed_preconditions=failures
        )


def _closed_jaw_targets(robot, context, joint_ids, arm_targets, side):
    arm = [joint_ids.index(index) for index in robot.get_joint_ids(f"{side}_arm")]
    jaws = [joint_ids.index(index) for index in robot.get_joint_ids(f"{side}_eef")]
    joints = context.robot.qpos[:, joint_ids]
    targets = joints[:, None].expand(-1, arm_targets.shape[1], -1).clone()
    targets[..., arm] = arm_targets
    targets[..., jaws] = 0.0
    return targets
