"""Check the pen's physical insertion after the native handover and placement."""

from dataclasses import replace
import math

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_skills import (
    GraspGoal,
    HeldObjectPoseGoal,
    JointPositionGoal,
    MoveHeldObject,
    MoveJoints,
    PickUp,
)
from embodichain.lab.sim.atomic_skills.affordance_sampling import AffordancePoseCandidates
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
from embodichain.utils.math import axis_angle_to_rotation_matrix

from .handover_targets import ReachableHandover, ReachableHandoverFactory
from .insertion_targets import (
    InsertionAlignmentFactory,
    InsertionLoweringFactory,
    InsertionPlaceFactory,
    _closed_jaw_targets,
    _shaft_in_gripper,
    cylinder_grasp_frame,
)
from .task_program_adapter import compose_task_program


def _pen_receive_frames(pen):
    """Grasp the exposed shaft from the receiving arm's side."""
    axis = pen[:, :3, 0]
    diagonal = pen.new_tensor((0, -math.sqrt(3) / 2, -0.5))
    approach = torch.nn.functional.normalize(
        diagonal - (axis * diagonal).sum(dim=-1, keepdim=True) * axis, dim=-1
    )
    grasp = torch.stack((axis, torch.linalg.cross(approach, axis), approach), dim=-1)
    slants = pen.new_tensor((0.0, -0.20, 0.20))
    rolls = axis_angle_to_rotation_matrix(
        slants[:, None] * pen.new_tensor((1, 0, 0))
    )
    flip = pen.new_tensor(((-1, 0, 0), (0, -1, 0), (0, 0, 1)))
    rotations = grasp[:, None] @ torch.cat((rolls, rolls @ flip))
    poses = pen[:, None].expand(-1, len(rolls) * 2, -1, -1).clone()
    poses[..., :3, :3] = rotations
    poses[..., :3, 3] += axis[:, None] * 0.016
    return poses


class PenPickup(RegisteredSemanticLowerer):
    """Reach the back half of the pen and leave its center open for transfer."""

    call_id = "robosyn.pick_pen_end"
    revision = "1"
    target_descriptor = PickUp.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH
    requires_unoccupied_slots = ("primary",)

    def __init__(self, robot, semantics):
        self.robot = robot
        self.semantics = semantics

    def lower(self, call, *, context, bound, option_template):
        pen = context.scene.entities["pen"].pose
        slants = pen.new_tensor(
            ((0, 0, 0), (0.15, 0, 0), (-0.15, 0, 0), (0, 0.28, 0), (0, -0.28, 0))
        )
        rotations = axis_angle_to_rotation_matrix(slants)
        flip = pen.new_tensor(((-1, 0, 0), (0, -1, 0), (0, 0, 1)))
        rotations = torch.cat((rotations, rotations @ flip))
        offsets = pen.new_tensor((-0.035, -0.047)).repeat_interleave(len(rotations))
        rotations = rotations.repeat(2, 1, 1)
        top_down = cylinder_grasp_frame(pen[:, :3, 0])
        grasps = pen[:, None].expand(-1, len(offsets), -1, -1).clone()
        grasps[..., :3, :3] = top_down[:, None] @ rotations
        grasps[..., :3, 3] += offsets[None, :, None] * pen[:, None, :3, 0]
        above = grasps.clone()
        above[..., :3, 3] -= grasps[..., :3, 2] * option_template.pre_grasp_distance
        lift = grasps.clone()
        lift[..., 2, 3] += option_template.lift_height
        path = torch.stack((above, grasps, lift), dim=2)
        arm = bound.binding.resources["primary"].endpoints["motion"].runtime_target
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        seeds = joints[:, None].expand(-1, path.shape[1] * path.shape[2], -1)
        valid, solutions = self.robot.compute_batch_ik(
            pose=path.flatten(1, 2), joint_seed=seeds, name=arm.control_part,
            env_ids=context.env_ids,
        )
        valid = valid.reshape(context.batch_size, len(offsets), 3).all(dim=-1)
        cost = (solutions - seeds).square().sum(dim=-1).reshape(*valid.shape, 3)
        cost = cost.sum(dim=-1).masked_fill(~valid, torch.inf)
        rows = torch.arange(context.batch_size, device=pen.device)
        return SemanticLowering(
            goal=GraspGoal(semantics=self.semantics, grasp_xpos=grasps[rows, cost.argmin(dim=1)]),
            failed_preconditions={
                int(context.env_ids[row]): "No reachable grasp leaving the pen center free."
                for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
            },
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id="pen",
                        slot_id="primary",
                    ),
                ),
            ),
        )


class PenPickupFactory:
    call_id = PenPickup.call_id
    revision = PenPickup.revision
    target_descriptor = PenPickup.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        pen = scene_registry.resolve("pen", expected_type=SceneObjectRef)
        grasp = scene_registry.resolve_affordance(
            pen, capability="affordance.grasp", explicit="pen_grasp"
        )
        return PenPickup(robot, scene_registry.object_semantics(pen, affordance=grasp))


class PenPresentation(RegisteredSemanticLowerer):
    """Present the pen from a source posture that also reaches the exchange."""

    call_id = "robosyn.present_pen"
    revision = "1"
    target_descriptor = MoveHeldObject.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot):
        self.robot = robot
        left = robot.get_link_pose("left_arm_base")[:, 1]
        right = robot.get_link_pose("right_arm_base")[:, 1]
        self.exchange_y = (left + right) / 2

    def lower(self, call, *, context, bound, option_template):
        pen = context.scene.entities["pen"].pose
        motion = bound.binding.resources["primary"].endpoints["motion"]
        arm = motion.runtime_target
        held = context.task.get_held_object(motion.task_state_key)
        if held is None or held.semantics.entity_id != "pen":
            raise ValueError("Pen presentation requires the source grasp.")

        positions = pen.new_tensor(
            [
                (x, -0.045, z)
                for x in (0.41, 0.445, 0.48, 0.515, 0.55)
                for z in (0.95, 0.975, 1.0, 1.025, 1.05)
            ]
        )
        yaw = pen.new_tensor((-0.09, -0.26, 0.09))
        roll = pen.new_tensor((0.0, -0.20, 0.20))
        rotation = (
            axis_angle_to_rotation_matrix(yaw[:, None] * pen.new_tensor((0, 0, 1)))[:, None]
            @ axis_angle_to_rotation_matrix(roll[:, None] * pen.new_tensor((1, 0, 0)))
        ).flatten(0, 1)
        count = len(positions) * len(rotation)
        targets = pen[:, None].expand(-1, count, -1, -1).clone()
        targets[..., :3, :3] = rotation.repeat(len(positions), 1, 1)
        targets[..., :3, 3] = positions.repeat_interleave(len(rotation), dim=0)
        exchange = targets.clone()
        exchange[..., 1, 3] = self.exchange_y[context.env_ids, None].to(pen)
        grasp = held.object_to_eef.to(pen).expand_as(pen)
        presentation = targets @ grasp[:, None]
        transfer = exchange @ grasp[:, None]
        retreat = transfer.clone()
        retreat[..., :3, 3] -= 0.012 * retreat[..., :3, 2]
        wrists = torch.stack((presentation, transfer, retreat), dim=2).flatten(1, 2)
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        seeds = joints[:, None].expand(-1, wrists.shape[1], -1)
        valid, solutions = self.robot.compute_batch_ik(
            pose=wrists, joint_seed=seeds, name=arm.control_part, env_ids=context.env_ids
        )
        valid = valid.reshape(context.batch_size, count, 3).all(dim=-1)
        cost = (solutions - seeds).square().sum(dim=-1).reshape(*valid.shape, 3)
        receive = _pen_receive_frames(exchange.flatten(0, 1)).reshape(
            context.batch_size, count, -1, 4, 4
        )
        pre_grasp = receive.clone()
        pre_grasp[..., :3, 3] -= 0.025 * receive[..., :3, 2]
        receive_path = torch.stack((pre_grasp, receive), dim=3).flatten(1, 3)
        left = context.robot.qpos[:, self.robot.get_joint_ids("left_arm")]
        receiver_valid, _ = self.robot.compute_batch_ik(
            pose=receive_path,
            joint_seed=left[:, None].expand(-1, receive_path.shape[1], -1),
            name="left_arm",
            env_ids=context.env_ids,
        )
        receiver_valid = receiver_valid.reshape(context.batch_size, count, -1, 2)
        receiver_ready = receiver_valid.all(dim=-1).any(dim=-1)
        opening = _PEN_PLACEMENT.opening_position(context.scene.entities["holder"].pose)
        position = exchange[..., :3, 3]
        far_from_holder = (position[..., 0] - opening[:, None, 0]).abs() > 0.12
        above_holder = position[..., 2] > opening[:, None, 2] + 0.070
        holder_outside_exchange = opening[:, None, 1] > position[..., 1] + 0.18
        clear_exchange = far_from_holder | above_holder | holder_outside_exchange
        use_clear_exchange = (clear_exchange & valid).any(dim=1, keepdim=True)
        clearance_penalty = (opening[:, None, 2] + 0.07 - position[..., 2]).clamp_min(0)
        costs = (
            cost.sum(dim=-1) + 30 * clearance_penalty + 3 * (~receiver_ready)
        ).masked_fill(
            ~valid | (use_clear_exchange & ~clear_exchange), torch.inf
        )
        rows = torch.arange(context.batch_size, device=pen.device)
        selected = costs.argmin(dim=1)
        # Keep the source jaw commanded closed all the way to the joint
        # workspace. HandOver can then receive the pen from a stationary wrist.
        waypoints = torch.stack((targets[rows, selected], exchange[rows, selected]), dim=1)
        return SemanticLowering(
            goal=HeldObjectPoseGoal(waypoints),
            failed_preconditions={
                int(context.env_ids[row]): "No pen presentation can reach the exchange."
                for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
            },
        )


class PenPresentationFactory:
    call_id = PenPresentation.call_id
    revision = PenPresentation.revision
    target_descriptor = PenPresentation.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return PenPresentation(robot)


class PenHandover(ReachableHandover):
    call_id = "robosyn.hand_over_pen"

    def extend_candidates(self, candidates, pen):
        poses = _pen_receive_frames(pen)
        max_cost = candidates.costs.masked_fill(~candidates.valid, 0).amax(dim=1)
        costs = (max_cost[:, None] + 0.2).expand_as(poses[..., 0, 0])
        return AffordancePoseCandidates(
            poses=torch.cat((candidates.poses, poses), dim=1),
            costs=torch.cat((candidates.costs, costs), dim=1),
            valid=torch.cat((candidates.valid, torch.ones_like(costs, dtype=torch.bool)), dim=1),
        )

    def rank_candidates(self, context, arm, candidates, pen, valid, cost, joints):
        # The source holds the negative end. Prefer exposed shaft on the
        # other side of the middle so the two closed jaws have room.
        poses = candidates.poses
        grip_offset = ((poses[..., :3, 3] - pen[:, None, :3, 3])
                       * pen[:, None, :3, 0]).sum(dim=-1)
        separated = valid & (grip_offset >= 0.008) & (grip_offset <= 0.026)
        eligible = torch.where(separated.any(dim=1, keepdim=True), separated, valid)
        cost = cost + 0.5 * ((grip_offset - 0.016) / 0.02).square()
        placement = replace(
            _PEN_PLACEMENT,
            yaw_candidates=_PEN_PLACEMENT.yaw_candidates[::2],
            tilt_candidates=(0.0, -0.17, 0.17, -0.34, 0.34),
        )
        return self.rank_insertion_candidates(
            context, arm, candidates, pen, eligible, cost, joints,
            eligible=eligible, placement=placement, clearance=_PEN_CLEARANCE,
            reach_margin_weight=0.3,
        )


class PenHandoverFactory(ReachableHandoverFactory):
    call_id = PenHandover.call_id
    object_id = "pen"
    lowerer_type = PenHandover


class PenClearanceLift(RegisteredSemanticLowerer):
    """Raise the received pen before swinging it toward the holder."""

    call_id = "robosyn.lift_received_pen"
    revision = "1"
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot):
        self.robot = robot

    def lower(self, call, *, context, bound, option_template):
        pen = context.scene.entities["pen"].pose
        holder = context.scene.entities["holder"].pose
        motion = bound.binding.resources["primary"].endpoints["motion"]
        arm_joints = self.robot.get_joint_ids("left_arm")
        qpos = context.robot.qpos[:, arm_joints]
        tcp = self.robot.compute_fk(
            qpos=qpos, name="left_arm", to_matrix=True, env_ids=context.env_ids
        )
        opening = _PEN_PLACEMENT.opening_position(holder)
        offsets = pen.new_tensor(
            ((-0.035, 0.01, 0.055), (-0.06, 0.02, 0.07), (0.0, 0.0, 0.06),
             (-0.07, 0.03, 0.045))
        )
        targets = pen[:, None].expand(-1, len(offsets), -1, -1).clone()
        targets[..., :3, 3] += offsets
        minimum_z = (opening[:, 2] + 0.09).minimum(pen[:, 2, 3] + 0.11)
        targets[..., 2, 3] = targets[..., 2, 3].maximum(minimum_z[:, None])
        actual_grasp = torch.linalg.solve(pen, tcp)
        pen_in_jaws = _shaft_in_gripper(actual_grasp)
        wrists = targets @ actual_grasp[:, None]
        valid, solutions = self.robot.compute_batch_ik(
            pose=wrists, joint_seed=qpos[:, None].expand(-1, len(offsets), -1),
            name="left_arm", env_ids=context.env_ids,
        )
        valid = valid.bool() & pen_in_jaws[:, None]
        costs = (solutions - qpos[:, None]).square().sum(dim=-1)
        selected = costs.masked_fill(~valid, torch.inf).argmin(dim=1)
        rows = torch.arange(context.batch_size, device=pen.device)
        path = _closed_jaw_targets(
            self.robot, context, list(motion.runtime_target.joint_ids),
            solutions[rows, selected][:, None], "left",
        )
        return SemanticLowering(
            goal=JointPositionGoal(path),
            failed_preconditions={
                int(context.env_ids[row]): (
                    "The receiver has lost the pen."
                    if not pen_in_jaws[row]
                    else "No reachable pose lifts the pen clear of the holder."
                )
                for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
            },
        )


class PenClearanceLiftFactory:
    call_id = PenClearanceLift.call_id
    revision = PenClearanceLift.revision
    target_descriptor = PenClearanceLift.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return PenClearanceLift(robot)


_PEN_CLEARANCE = 0.003
_PEN_PLACEMENT = InsertionPlaceFactory(
    "pen",
    "holder",
    (0.0, 0.136, 0.0),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    above_opening=0.058,
    yaw_candidates=tuple(index * math.pi / 16 for index in range(32)),
    tilt_candidates=(0.0, -0.17, 0.17, -0.34, 0.34, -0.45, 0.45, -0.57, 0.57),
    insertion_axis=(1, 0, 0),
    opening_offset=(0, 0.096, 0),
)
_DEPLOYMENT, _ADAPTER = compose_task_program(
    "items_handover",
    (replace(_PEN_PLACEMENT, release_at_current_pose=True),),
    additional_lowerers=(
        PenPickupFactory(),
        PenPresentationFactory(),
        PenHandoverFactory(),
        PenClearanceLiftFactory(),
        InsertionAlignmentFactory(
            _PEN_PLACEMENT, lift_height=_PEN_CLEARANCE, preorient=True
        ),
        InsertionLoweringFactory(_PEN_PLACEMENT),
    ),
    use_scene_grasp_options=True,
)
_PROGRAM = load_task_program(
    _DEPLOYMENT.program_path,
    integration=_DEPLOYMENT.selection,
    validation_context=_ADAPTER.registration.catalog,
)


@register_env(
    "RoboSyn-ItemsHandover-v1",
    max_episode_steps=1500,
    task_program_registration=_ADAPTER.registration,
    task_program_adapter_factory=_ADAPTER,
)
class ItemsHandover(EmbodiedEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.task_program is None:
            cfg.task_program = _PROGRAM
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
