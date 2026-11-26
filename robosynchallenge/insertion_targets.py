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

    call_id = "simulation.place_relative"
    revision = "1"
    target_descriptor = Place.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        return InsertionPlace(robot, self)


class InsertionPlace(RegisteredSemanticLowerer):
    call_id = InsertionPlaceFactory.call_id
    target_descriptor = InsertionPlaceFactory.target_descriptor
    effect_contract_kind = SemanticEffectKind.RELEASE

    def __init__(self, robot, placement):
        self.robot = robot
        self.placement = placement

    def select_wrist_path(self, context, control_part, lift_height):
        placement = self.placement
        joints = context.robot.qpos[:, self.robot.get_joint_ids(control_part)]
        tcp = self.robot.compute_fk(
            qpos=joints, name=control_part, to_matrix=True, env_ids=context.env_ids
        )
        object_pose = context.scene.entities[placement.object_id].pose
        receptacle = context.scene.entities[placement.receptacle_id].pose
        object_to_tcp = torch.linalg.solve(object_pose, tcp)

        # Rotating around vertical leaves the insertion axis and drop point fixed.
        yaw = tcp.new_tensor(placement.yaw_candidates).repeat_interleave(
            len(placement.tilt_candidates)
        )
        pitch = tcp.new_tensor(placement.tilt_candidates).repeat(
            len(placement.yaw_candidates)
        )
        yaw_rotation = axis_angle_to_rotation_matrix(
            yaw[:, None] * tcp.new_tensor((0.0, 0.0, 1.0))
        )
        orientation = (
            object_pose[:, None, :3, :3]
            if placement.preserve_object_rotation
            else tcp.new_tensor(placement.upright_rotation)
        )
        pitch_rotation = axis_angle_to_rotation_matrix(
            pitch[:, None] * tcp.new_tensor((0.0, 1.0, 0.0))
        )
        rotation = yaw_rotation @ pitch_rotation @ orientation
        targets = receptacle[:, None].expand(-1, len(yaw), -1, -1).clone()
        if placement.follow_receptacle_vertical:
            rotation = receptacle[:, None, :3, :3] @ rotation
        targets[..., :3, :3] = rotation
        offset = tcp.new_tensor(placement.release_offset)
        if not placement.offset_in_world:
            offset = receptacle[:, :3, :3] @ offset
        targets[..., :3, 3] = (receptacle[:, :3, 3] + offset)[:, None]
        release_tcp = targets @ object_to_tcp[:, None]
        above_tcp = release_tcp.clone()
        above_tcp[..., 2, 3] += placement.above_opening
        clearance_tcp = above_tcp.clone()
        clearance_tcp[..., 2, 3] += lift_height
        retract_tcp = release_tcp.clone()
        retract_tcp[..., 2, 3] += lift_height
        poses = torch.stack(
            (clearance_tcp, above_tcp, release_tcp, retract_tcp), dim=2
        ).flatten(1, 2)
        seeds = joints[:, None].expand(-1, poses.shape[1], -1)
        valid, solutions = self.robot.compute_batch_ik(
            pose=poses,
            joint_seed=seeds,
            name=control_part,
            env_ids=context.env_ids,
        )
        valid = valid.reshape(context.batch_size, len(yaw), 4).all(dim=-1)
        costs = (
            (solutions - seeds)
            .square()
            .sum(dim=-1)
            .reshape(context.batch_size, len(yaw), 4)
            .sum(dim=-1)
            .masked_fill(~valid, torch.inf)
        )
        upright = pitch == 0
        upright_available = (valid & upright).any(dim=1, keepdim=True)
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
            for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
        }
        joint_path = solutions.reshape(context.batch_size, len(yaw), 4, -1)[
            rows, chosen
        ]
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

    call_id = "robosyn.align_for_insertion"
    revision = "1"
    target_descriptor = MoveJoints.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        return InsertionAlignment(InsertionPlace(robot, self.placement))


class InsertionAlignment(RegisteredSemanticLowerer):
    call_id = InsertionAlignmentFactory.call_id
    target_descriptor = InsertionAlignmentFactory.target_descriptor
    preserves_symbolic_state = True

    def __init__(self, placement):
        self.placement = placement

    def lower(self, call, *, context, bound, option_template):
        robot = self.placement.robot
        motion = bound.binding.resources["primary"].endpoints["motion"]
        joint_ids = list(motion.runtime_target.joint_ids)
        side = self.placement.placement.arm_side
        _, joints, failures = self.placement.select_wrist_path(
            context, f"{side}_arm", 0.025
        )
        path = _closed_jaw_targets(robot, context, joint_ids, joints[:, (0, 1)], side)
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
        current_axis = held_object[:, :3, 2]
        rack_axis = receptacle[:, :3, 2]
        axis = torch.linalg.cross(current_axis, rack_axis)
        norm = axis.norm(dim=1)
        angle = torch.atan2(norm, (current_axis * rack_axis).sum(dim=1))
        correction = axis * (angle / norm.clamp_min(1e-6))[:, None]
        rotations = axis_angle_to_rotation_matrix(correction)
        object_targets = held_object[:, None].expand(-1, 3, -1, -1).clone()
        object_targets[..., :3, :3] = (rotations @ held_object[:, :3, :3])[:, None]
        heights = tcp.new_tensor(
            (placement.above_opening, placement.above_opening / 2, 0)
        )
        object_targets[..., :3, 3] = (
            destination[:, None] + heights[None, :, None] * rack_axis[:, None]
        )
        wrist_targets = object_targets @ torch.linalg.solve(held_object, tcp)[:, None]
        valid, wrist = robot.compute_batch_ik(
            pose=wrist_targets,
            joint_seed=arm_joints[:, None].expand(-1, 3, -1),
            name=control_part,
            env_ids=context.env_ids,
        )
        failures = {
            int(
                context.env_ids[row]
            ): "The aligned insertion path is outside the arm's reach."
            for row in (~valid.all(dim=1)).nonzero().flatten().tolist()
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
