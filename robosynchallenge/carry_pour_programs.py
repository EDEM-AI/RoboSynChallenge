"""Measure physical task outcomes for the native carry and pour programs."""

from dataclasses import dataclass
import math

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_skills import (
    GraspGoal,
    HeldObjectPoseGoal,
    MoveHeldObject,
    PickUp,
)
from embodichain.lab.task_program.semantics import SceneObjectRef
from embodichain.lab.task_program.compiler.lowering import (
    HeldObjectRelation,
    RegisteredHeldObjectEffect,
    RegisteredSemanticEffect,
    RegisteredSemanticLowerer,
    SemanticEffectKind,
    SemanticLowering,
)
from embodichain.lab.task_program.language.loader import load_task_program
from embodichain.utils.math import axis_angle_to_rotation_matrix

from .insertion_targets import InsertionPlaceFactory
from .task_program_adapter import InvocationMotionPolicyEnv, compose_task_program

_POUR_ANGLE = -1.10
_BOTTLE_MOUTH_HEIGHT = 0.236


def _pour_arc(cup):
    """Upright, midway and tilted poses that aim the bottle mouth at the cup."""
    # The approach can come from either side of the cup. Limiting yaw to
    # the cup's front arc leaves some valid bottle grasps with no pour pose.
    headings = torch.arange(-12, 12, device=cup.device, dtype=cup.dtype)
    headings *= math.pi / 12
    heights = cup.new_tensor((0.045, 0.065, 0.085, 0.105, 0.125))
    yaw = headings.repeat(len(heights))
    roll = cup.new_tensor((1.0, 0.0, 0.0))
    up = cup.new_tensor((0.0, 0.0, 1.0))
    rotation = axis_angle_to_rotation_matrix(yaw[:, None] * up)
    upright = rotation @ axis_angle_to_rotation_matrix(roll * (math.pi / 2))
    tilted = upright @ axis_angle_to_rotation_matrix(roll * _POUR_ANGLE)
    # The in-place Pour skill rotates about the bottle's authored local X.
    mouth = cup.new_tensor((0.0, _BOTTLE_MOUTH_HEIGHT, 0.0))
    mouth_at_tilt = tilted @ mouth
    offsets = heights.repeat_interleave(len(headings))[:, None] * up
    pivot = cup[:, None, :3, 3] + offsets
    pivot[..., :2] -= mouth_at_tilt[:, :2]
    targets = cup[:, None].expand(-1, len(yaw), -1, -1).clone()
    targets[..., :3, :3] = upright
    targets[..., :3, 3] = pivot
    mid_tilt = upright @ axis_angle_to_rotation_matrix(roll * (_POUR_ANGLE / 2))
    pour_path = targets[:, :, None].expand(-1, -1, 3, -1, -1).clone()
    pour_path[:, :, 1, :3, :3] = mid_tilt
    pour_path[:, :, 2, :3, :3] = tilted
    return pour_path, (1 - headings.cos()).repeat(len(heights))


class BottlePickup(RegisteredSemanticLowerer):
    """Approach upright bottles at the body and fallen bottles from above."""

    call_id = "robosyn.pick_bottle"
    revision = "1"
    target_descriptor = PickUp.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH
    requires_unoccupied_slots = ("primary",)

    def __init__(self, semantics, robot, screen_pour, grasp_generator):
        self.semantics = semantics
        self.robot = robot
        self.right_base = robot.get_link_pose("right_arm_base")[:, :3].clone()
        self.screen_pour = screen_pour
        self.grasp_generator = grasp_generator

    def _screen_grasps(self, bottle, center, direction, context, bound, options):
        yaw = bottle.new_tensor((0.0, -0.52, 0.52, -1.05, 1.05, -1.57, 1.57))
        height = bottle.new_tensor((0.0, -0.016, 0.016)).repeat_interleave(len(yaw))
        yaw = yaw.repeat(3)
        turn = axis_angle_to_rotation_matrix(yaw[:, None] * bottle.new_tensor((0, 0, 1)))
        approach = (turn[None] @ direction[:, None, :, None]).squeeze(-1)
        approach[..., 2] = -0.25
        approach = torch.nn.functional.normalize(approach, dim=-1)
        axis = bottle[:, None, :3, 1].expand_as(approach)
        # Approach a bottle lying on its side from above. A tall upright
        # bottle is gripped across the barrel from the robot's side.
        if options.approach_along_grasp_axis:
            approach = torch.where(
                axis[..., 2:].abs() > 0.85,
                approach,
                bottle.new_tensor((0.0, 0.0, -1.0)),
            )
        lateral = torch.nn.functional.normalize(
            torch.linalg.cross(approach, axis), dim=-1
        )
        grasps = bottle[:, None].expand(-1, len(yaw), -1, -1).clone()
        grasps[..., :3, 0] = torch.linalg.cross(lateral, approach)
        grasps[..., :3, 1] = lateral
        grasps[..., :3, 2] = approach
        grasps[..., :3, 3] = center[:, None] + height[:, None] * axis
        valid_geometry = torch.ones(
            grasps.shape[:2], device=grasps.device, dtype=torch.bool
        )
        rank_bias = grasps.new_zeros(grasps.shape[:2])
        if options.approach_along_grasp_axis:
            # Use the calibrated native gripper to find opposing contact
            # patches on the waist before falling back to analytic grasps.
            candidates = self.semantics.affordance.get_grasp_candidates(
                self.grasp_generator, bottle, approach[:, 0]
            )
            along_barrel = (
                (candidates.poses[..., :3, 3] - center[:, None])
                * bottle[:, None, :3, 1]
            ).sum(dim=-1)
            suitable = candidates.valid & (along_barrel.abs() <= 0.043)
            count = min(12, candidates.poses.shape[1])
            shortlist = candidates.costs.masked_fill(~suitable, torch.inf).topk(
                count, dim=1, largest=False
            ).indices
            rows = torch.arange(context.batch_size, device=grasps.device)[:, None]
            native_valid = suitable[rows, shortlist]
            native = torch.where(
                native_valid[..., None, None],
                candidates.poses[rows, shortlist],
                grasps[:, :1],
            )
            grasps = torch.cat((native, grasps), dim=1)
            valid_geometry = torch.cat((native_valid, valid_geometry), dim=1)
            rank_bias = torch.cat(
                (grasps.new_zeros((context.batch_size, count)), rank_bias + 1.0), dim=1
            )

        pre_grasp = grasps.clone()
        pickup_direction = grasps[..., :3, 2] if options.approach_along_grasp_axis else (
            torch.nn.functional.normalize(options.approach_direction.to(bottle), dim=-1)
            .reshape(-1, 1, 3)
        )
        pre_grasp[..., :3, 3] -= pickup_direction * options.pre_grasp_distance
        lift = grasps.clone()
        lift[..., 2, 3] += options.lift_height
        waypoints = torch.stack((pre_grasp, grasps, lift), dim=2)
        arm = bound.binding.resources["primary"].endpoints["motion"].runtime_target
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        grasp_count = grasps.shape[1]
        if options.cartesian_lift and options.approach_along_grasp_axis:
            # Endpoints alone can straddle an IK singularity. Screen the
            # same vertical lift before handing the grasp to native PickUp.
            lift_steps = max(2, math.ceil(options.lift_height / 0.004))
            heights = torch.linspace(
                0, options.lift_height, lift_steps + 1,
                dtype=bottle.dtype, device=bottle.device,
            )[1:]
            lift_path = grasps[:, :, None].expand(-1, -1, lift_steps, -1, -1).clone()
            lift_path[..., 2, 3] += heights
            waypoints = torch.cat((waypoints[:, :, :2], lift_path), dim=2)
            valid, qpos = self.robot.compute_batch_ik(
                pose=waypoints.flatten(0, 1),
                joint_seed=joints.repeat_interleave(grasp_count, dim=0),
                name=arm.control_part,
                env_ids=context.env_ids.repeat_interleave(grasp_count),
                continuous=True,
            )
            smooth_lift = qpos[:, 1:].diff(dim=1).abs().amax(dim=(1, 2)) <= 0.09
            valid_geometry &= smooth_lift.reshape(context.batch_size, grasp_count)
        else:
            poses = waypoints.flatten(1, 2)
            valid, qpos = self.robot.compute_batch_ik(
                pose=poses,
                joint_seed=joints[:, None].expand(-1, poses.shape[1], -1),
                name=arm.control_part,
                env_ids=context.env_ids,
            )
        valid = valid.reshape(context.batch_size, grasp_count, -1).all(dim=2)
        valid &= valid_geometry
        qpos = qpos.reshape(context.batch_size, grasp_count, waypoints.shape[2], -1)
        start_cost = (qpos[:, :, 0] - joints[:, None]).square().sum(dim=-1)
        motion_cost = qpos.diff(dim=2).square().sum(dim=(-1, -2))
        pour_path, side_cost = _pour_arc(context.scene.entities["cup"].pose)
        pour_count = pour_path.shape[1]
        object_to_tcp = torch.linalg.solve(bottle[:, None], grasps)
        pour_poses = (pour_path[:, None] @ object_to_tcp[:, :, None, None]).flatten(1, 3)
        lift_joints = qpos[:, :, -1]
        pour_seeds = lift_joints[:, :, None, None].expand(-1, -1, pour_count, 3, -1)
        can_pour, pour_joints = self.robot.compute_batch_ik(
            pose=pour_poses,
            joint_seed=pour_seeds.flatten(1, 3),
            name=arm.control_part,
            env_ids=context.env_ids,
        )
        pour_plan_shape = (context.batch_size, grasp_count, pour_count, 3)
        can_pour = can_pour.reshape(pour_plan_shape).all(dim=-1)
        pour_joints = pour_joints.reshape(*pour_plan_shape, -1)
        transfer_cost = (
            (pour_joints[..., 0, :] - lift_joints[:, :, None, :]).square().sum(dim=-1)
        )
        tilt_cost = pour_joints.diff(dim=3).square().sum(dim=(-1, -2))
        pour_cost = (transfer_cost + 2 * tilt_cost + side_cost).masked_fill(
            ~can_pour, torch.inf
        )
        valid_for_pour = valid & can_pour.any(dim=2)
        has_pour_plan = valid_for_pour.any(dim=1, keepdim=True)
        travel_cost = start_cost + motion_cost + rank_bias
        cost = torch.where(
            has_pour_plan,
            (travel_cost + pour_cost.amin(dim=2)).masked_fill(~valid_for_pour, torch.inf),
            travel_cost.masked_fill(~valid, torch.inf),
        )
        rows = torch.arange(context.batch_size, device=bottle.device)
        return grasps[rows, cost.argmin(dim=1)], valid.any(dim=1)

    def lower(self, call, *, context, bound, option_template):
        bottle = context.scene.entities["bottle"].pose
        axis = bottle[:, :3, 1]
        effect = RegisteredSemanticEffect(
            effect_kind=SemanticEffectKind.ATTACH,
            held_objects=(
                RegisteredHeldObjectEffect(
                    expectation_id="primary",
                    relation=HeldObjectRelation.ATTACHED,
                    object_id="bottle",
                    slot_id="primary",
                ),
            ),
        )
        if not option_template.approach_along_grasp_axis and not bool((axis[:, 2] > 0.90).all()):
            return SemanticLowering(
                goal=GraspGoal(semantics=self.semantics), registered_effect=effect
            )
        center = bottle[:, :3, 3] + 0.105 * axis
        direction = center - self.right_base[context.env_ids].to(center)
        direction[:, 2] = 0.0
        direction = torch.nn.functional.normalize(direction, dim=1)
        if self.screen_pour:
            wrist, feasible = self._screen_grasps(
                bottle, center, direction, context, bound, option_template
            )
            return SemanticLowering(
                goal=GraspGoal(semantics=self.semantics, grasp_xpos=wrist),
                failed_preconditions={
                    int(context.env_ids[row]): "No pickup pose for the pour arc."
                    for row in (~feasible).nonzero().flatten().tolist()
                },
                registered_effect=effect,
            )
        direction[:, 2] = -0.25
        approach = torch.nn.functional.normalize(direction, dim=1)
        lateral = torch.nn.functional.normalize(
            torch.linalg.cross(approach, axis), dim=1
        )
        wrist = bottle.clone()
        wrist[:, :3, 0] = torch.linalg.cross(lateral, approach)
        wrist[:, :3, 1] = lateral
        wrist[:, :3, 2] = approach
        wrist[:, :3, 3] = center
        return SemanticLowering(
            goal=GraspGoal(semantics=self.semantics, grasp_xpos=wrist),
            registered_effect=effect,
        )


@dataclass(frozen=True)
class BottlePickupFactory:
    call_id = BottlePickup.call_id
    revision = BottlePickup.revision
    target_descriptor = BottlePickup.target_descriptor

    screen_pour: bool = False

    def create(self, *, simulation, robot, scene_registry, engine):
        bottle = scene_registry.resolve("bottle", expected_type=SceneObjectRef)
        affordance = scene_registry.resolve_affordance(
            bottle, capability="affordance.grasp", explicit="bottle_grasp"
        )
        return BottlePickup(
            scene_registry.object_semantics(bottle, affordance=affordance),
            robot,
            self.screen_pour,
            engine.grasp_pose_generators.get("right_eef"),
        )


class BottlePourPosition(RegisteredSemanticLowerer):
    """Find an upright bottle pose whose in-place tilt reaches the cup."""

    call_id = "robosyn.position_bottle_to_pour"
    revision = "1"
    target_descriptor = MoveHeldObject.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH

    def __init__(self, robot, lift_before_transport, keep_to_holding_side):
        self.robot = robot
        self.lift_before_transport = lift_before_transport
        self.keep_to_holding_side = keep_to_holding_side

    def lower(self, call, *, context, bound, option_template):
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        arm = endpoint.runtime_target
        held = context.task.get_held_object(endpoint.task_state_key)
        if held is None or held.semantics.entity_id != "bottle":
            raise ValueError("Pour positioning requires the held bottle.")
        cup = context.scene.entities["cup"].pose
        bottle = context.scene.entities["bottle"].pose
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        pour_path, approach_side_cost = _pour_arc(cup)
        targets = pour_path[:, :, 0]
        candidate_count = pour_path.shape[1]
        wrist = self.robot.compute_fk(
            qpos=joints, name=arm.control_part, to_matrix=True, env_ids=context.env_ids
        )
        observed = torch.linalg.solve(bottle, wrist)
        planned = held.object_to_eef.to(cup).expand_as(bottle)
        via = targets.clone()
        via[..., 1, 3] = bottle[:, None, 1, 3]
        via[..., 2, 3] = torch.maximum(
            targets[..., 2, 3], bottle[:, None, 2, 3]
        )
        if self.lift_before_transport:
            via[..., 0, 3] = via[..., 0, 3].clamp_min(0.53)
        full_path = torch.cat((via[:, :, None], pour_path), dim=2)
        if self.lift_before_transport:
            # Lift from the observed pose before turning toward the cup.
            lift = bottle[:, None].expand_as(targets).clone()
            lift[..., 2, 3] += 0.020
            full_path = torch.cat((lift[:, :, None], full_path), dim=2)
        poses = (full_path @ observed[:, None, None]).flatten(1, 2)
        seeds = joints[:, None].expand(-1, poses.shape[1], -1)
        valid, qpos = self.robot.compute_batch_ik(
            pose=poses, joint_seed=seeds, name=arm.control_part, env_ids=context.env_ids
        )
        path_length = full_path.shape[2]
        valid = valid.reshape(context.batch_size, candidate_count, path_length).all(dim=-1)
        qpos = qpos.reshape(context.batch_size, candidate_count, path_length, -1)
        approach_cost = (qpos[:, :, 0] - joints[:, None]).square().sum(dim=-1)
        tilt_cost = qpos.diff(dim=2).square().sum(dim=(-1, -2))
        # Prefer keeping the bottle on the side of the arm holding it.
        cost = (approach_cost + 2 * tilt_cost + approach_side_cost).masked_fill(
            ~valid, torch.inf
        )
        if self.keep_to_holding_side:
            # Avoid passing the bottle across the cup or turning its barrel
            # unnecessarily inside the jaws.
            grip_rotation = (targets[..., :3, 0] * bottle[:, None, :3, 0]).sum(dim=-1)
            cost = cost + 4 * (1 - grip_rotation)
            right_side = targets[..., 1, 3] < cup[:, None, 1, 3] - 0.035
            right_side &= (targets[..., 0, 3] >= 0.49) | (targets[..., 1, 3] >= -0.18)
            use_right_side = (right_side & valid).any(dim=1, keepdim=True)
            cost = cost.masked_fill(use_right_side & ~right_side, torch.inf)
        rows = torch.arange(context.batch_size, device=cup.device)
        selected = full_path[rows, cost.argmin(dim=1), :-2]
        selected = selected @ (observed @ torch.linalg.inv(planned))[:, None]
        failures = {
            int(
                context.env_ids[row]
            ): "No reachable upright and tilted bottle pose at the cup."
            for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
        }
        return SemanticLowering(
            goal=HeldObjectPoseGoal(selected, object_id="bottle"),
            failed_preconditions=failures,
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id="bottle",
                        slot_id="primary",
                        retain_during_segments=("transport",),
                    ),
                ),
            ),
        )


@dataclass(frozen=True)
class BottlePourPositionFactory:
    call_id = BottlePourPosition.call_id
    revision = BottlePourPosition.revision
    target_descriptor = BottlePourPosition.target_descriptor
    lift_before_transport: bool = False
    keep_to_holding_side: bool = False

    def create(self, *, simulation, robot, scene_registry, engine):
        return BottlePourPosition(
            robot, self.lift_before_transport, self.keep_to_holding_side
        )


class BasketHandlePickup(RegisteredSemanticLowerer):
    """Bind the authored handle arc to the native rigid-object PickUp skill."""

    call_id = "robosyn.pick_basket_handle"
    revision = "1"
    target_descriptor = PickUp.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH
    requires_unoccupied_slots = ("primary",)

    def __init__(self, semantics):
        self.semantics = semantics

    def lower(self, call, *, context, bound, option_template):
        basket = context.scene.entities["basket"].pose
        tilt = math.sqrt(0.5)
        local_grasp = basket.new_tensor(
            ((tilt, 0, tilt, 0), (tilt, 0, -tilt, 0.169), (0, 1, 0, 0), (0, 0, 0, 1))
        )
        return SemanticLowering(
            goal=GraspGoal(semantics=self.semantics, grasp_xpos=basket @ local_grasp),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id="basket",
                        slot_id="primary",
                    ),
                ),
            ),
        )


class BasketHandlePickupFactory:
    call_id = BasketHandlePickup.call_id
    revision = BasketHandlePickup.revision
    target_descriptor = BasketHandlePickup.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        basket = scene_registry.resolve("basket", expected_type=SceneObjectRef)
        affordance = scene_registry.resolve_affordance(
            basket, capability="affordance.grasp", explicit="basket_grasp"
        )
        return BasketHandlePickup(
            scene_registry.object_semantics(basket, affordance=affordance)
        )


class BasketCarry(RegisteredSemanticLowerer):
    """Select a reachable transfer that keeps the loaded basket level."""

    call_id = "simulation.move_held_object"
    revision = "1"
    target_descriptor = MoveHeldObject.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot):
        self.robot = robot

    def lower(self, call, *, context, bound, option_template):
        table = context.scene.entities["table"].pose
        basket = context.scene.entities["basket"].pose
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        arm = endpoint.runtime_target
        held = context.task.get_held_object(endpoint.task_state_key)
        if held is None or held.semantics.entity_id != "basket":
            raise ValueError("Basket transfer requires the handle grasp.")
        offsets = table.new_tensor(
            [
                (x, y, height)
                for x in (-0.125, -0.17, -0.21, -0.255)
                for y in (0.065, 0.085, 0.11)
                for height in (0.13, 0.16)
            ]
        )
        targets = basket[:, None].expand(-1, len(offsets), -1, -1).clone()
        targets[..., :3, 3] = table[:, None, :3, 3] + offsets
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        wrist = self.robot.compute_fk(
            qpos=joints, name=arm.control_part, to_matrix=True, env_ids=context.env_ids
        )
        observed_relation = torch.linalg.solve(basket, wrist)
        planned_relation = held.object_to_eef.to(basket)
        if planned_relation.ndim == 2:
            planned_relation = planned_relation.expand(context.batch_size, -1, -1)
        # Approach the destination from the same side before translating left.
        near = targets.clone()
        near[..., 1, 3] = basket[:, None, 1, 3]
        actual_targets = torch.stack((near, targets), dim=2)
        wrist_targets = actual_targets @ observed_relation[:, None, None]
        poses = wrist_targets.flatten(1, 2)
        seeds = joints[:, None].expand(-1, poses.shape[1], -1)
        success, solutions = self.robot.compute_batch_ik(
            pose=poses, joint_seed=seeds, name=arm.control_part, env_ids=context.env_ids
        )
        layout = (context.batch_size, len(offsets), 2)
        valid = success.reshape(layout).all(dim=-1)
        costs = (
            (solutions - seeds)
            .square()
            .sum(dim=-1)
            .reshape(layout)
            .sum(dim=-1)
            .masked_fill(~valid, torch.inf)
        )
        rows = torch.arange(context.batch_size, device=basket.device)
        selected = wrist_targets[rows, costs.argmin(dim=-1)] @ (
            torch.linalg.inv(planned_relation)[:, None]
        )
        failures = {
            int(
                context.env_ids[row]
            ): "No level carry pose is reachable with the handle grasp."
            for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
        }
        return SemanticLowering(
            goal=HeldObjectPoseGoal(selected), failed_preconditions=failures
        )


class BasketCarryFactory:
    call_id = BasketCarry.call_id
    revision = BasketCarry.revision
    target_descriptor = BasketCarry.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return BasketCarry(robot)


_BASKET_DEPLOYMENT, _BASKET_ADAPTER = compose_task_program(
    "handle_basket",
    (
        InsertionPlaceFactory(
            "milk",
            "basket",
            (0.0, 0.058, 0.136),
            ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
            offset_in_world=True,
            yaw_candidates=tuple(index * math.pi / 8 for index in range(16)),
            tilt_candidates=(0.0, -0.26, 0.26),
        ),
        BasketCarryFactory(),
    ),
    additional_lowerers=(BasketHandlePickupFactory(),),
    motion_presets={"clear_left_arm": {"sample_count": 50}},
)
_BASKET_PROGRAM = load_task_program(
    _BASKET_DEPLOYMENT.program_path,
    integration=_BASKET_DEPLOYMENT.selection,
    validation_context=_BASKET_ADAPTER.registration.catalog,
)
def _water_pour_deployment(*, lift_above_distractors=False):
    integration_overrides = None
    if lift_above_distractors:
        # In the cluttered scene, reach along the selected grasp axis and
        # lift above the nearby objects before turning the bottle.
        integration_overrides = {
            "profile": {
                "skill_options": {
                    BottlePickup.call_id: {
                        "pre_grasp_distance": 0.044,
                        "lift_height": 0.10,
                    }
                }
            }
        }
    deployment, adapter = compose_task_program(
        "water_pouring",
        (),
        additional_lowerers=(
            BottlePickupFactory(screen_pour=True),
            BottlePourPositionFactory(
                lift_before_transport=lift_above_distractors,
                keep_to_holding_side=True,
            ),
        ),
        integration_overrides=integration_overrides,
    )
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=adapter.registration.catalog,
    )
    return adapter, program


_POUR_ADAPTER, _POUR_PROGRAM = _water_pour_deployment()
_CLUTTERED_POUR_ADAPTER, _CLUTTERED_POUR_PROGRAM = _water_pour_deployment(
    lift_above_distractors=True
)


@register_env(
    "RoboSyn-HandleBasket-v1",
    max_episode_steps=1500,
    task_program_registration=_BASKET_ADAPTER.registration,
    task_program_adapter_factory=_BASKET_ADAPTER,
)
class HandleBasket(InvocationMotionPolicyEnv):
    call_motion_presets = {
        (_BASKET_PROGRAM.program_id, "simulation.park"): "clear_left_arm"
    }
    def __init__(self, cfg, **kwargs):
        if cfg.task_program is None:
            cfg.task_program = _BASKET_PROGRAM
        super().__init__(cfg=cfg, **kwargs)
        self._milk = self.sim.get_rigid_object("milk")
        self._basket = self.sim.get_rigid_object("basket")
        self._initial_y = torch.zeros(self.num_envs, device=self.device)
        self._stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        rows = (options or {}).get("reset_ids", slice(None))
        self._initial_y[rows] = self._basket.get_local_pose()[rows, 1]
        self._stable_steps[rows] = 0
        return obs, info

    def compute_task_state(self, **kwargs):
        basket = self._basket.get_local_pose()
        milk = self._milk.get_local_pose()
        distance = torch.linalg.vector_norm(milk[:, :2] - basket[:, :2], dim=-1)
        displacement = basket[:, 1] - self._initial_y
        in_basket = (distance < 0.10) & (milk[:, 2] > basket[:, 2])
        holding_goal = in_basket & (displacement > 0.15)
        self._stable_steps = torch.where(holding_goal, self._stable_steps + 1, 0)
        keep_running = torch.zeros_like(holding_goal)
        return (
            keep_running,
            keep_running,
            {
                "milk_basket_distance_m": distance,
                "basket_displacement_y_m": displacement,
                "stable_steps": self._stable_steps,
            },
        )

    def is_task_success(self, **kwargs):
        return self._stable_steps >= 55


@register_env(
    "RoboSyn-WaterPouring-v1",
    max_episode_steps=1500,
    task_program_registration=_POUR_ADAPTER.registration,
    task_program_adapter_factory=_POUR_ADAPTER,
)
class WaterPouring(EmbodiedEnv):
    program = _POUR_PROGRAM

    def __init__(self, cfg, **kwargs):
        if (cfg.extensions or {}).get("lift_bottle_above_distractors", False):
            if cfg.task_program is None:
                cfg.task_program = _CLUTTERED_POUR_PROGRAM
            adapter = kwargs.get("task_program_adapter_factory")
            if adapter is None or adapter is _POUR_ADAPTER:
                kwargs["task_program_adapter_factory"] = _CLUTTERED_POUR_ADAPTER
        if cfg.task_program is None:
            cfg.task_program = self.program
        super().__init__(cfg=cfg, **kwargs)
        self._bottle = self.sim.get_rigid_object("bottle")
        self._cup = self.sim.get_rigid_object("cup")
        self._initial_bottle_z = torch.zeros(self.num_envs, device=self.device)
        self._grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lost_grasp = torch.zeros_like(self._grasped)
        self._poured = torch.zeros_like(self._grasped)
        self._success = torch.zeros_like(self._grasped)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        rows = (options or {}).get("reset_ids", slice(None))
        self._initial_bottle_z[rows] = self._bottle.get_local_pose()[rows, 2]
        self._grasped[rows] = False
        self._lost_grasp[rows] = False
        self._poured[rows] = False
        self._success[rows] = False
        return obs, info

    def compute_task_state(self, **kwargs):
        bottle = self._bottle.get_local_pose(to_matrix=True)
        cup = self._cup.get_local_pose(to_matrix=True)
        eef = self.robot.get_link_pose("right_link6")
        axis = bottle[:, :3, 1]
        bottle_position = bottle[:, :3, 3]
        cup_position = cup[:, :3, 3]
        angle = torch.acos(axis[:, 2].clamp(-1.0, 1.0))
        mouth_to_cup = cup_position - (bottle_position + _BOTTLE_MOUTH_HEIGHT * axis)
        distance = torch.linalg.vector_norm(mouth_to_cup[:, :2], dim=-1)
        height = -mouth_to_cup[:, 2]
        near_gripper = (
            torch.linalg.vector_norm(bottle_position - eef[:, :3], dim=-1) < 0.30
        )
        held = near_gripper & (
            self._grasped | (bottle_position[:, 2] > self._initial_bottle_z + 0.03)
        )
        cup_upright = cup[:, 2, 2] > math.cos(math.pi / 4)
        mouth_over_cup = (
            (distance < 0.08)
            & (height > 0.04)
            & (height < 0.30)
            & ((axis[:, :2] * mouth_to_cup[:, :2]).sum(dim=-1) > -0.02)
        )
        self._grasped |= held
        self._lost_grasp |= self._grasped & ~held
        self._poured |= (
            held
            & cup_upright
            & mouth_over_cup
            & (angle > math.pi / 4)
            & (angle < 2 * math.pi / 3)
        )
        self._success |= (
            self._poured
            & held
            & ~self._lost_grasp
            & cup_upright
            & (angle < math.pi / 4)
        )
        keep_running = torch.zeros_like(held)
        return (
            keep_running,
            keep_running,
            {
                "bottle_angle_rad": angle,
                "bottle_mouth_cup_distance_m": distance,
                "bottle_mouth_cup_height_m": height,
                "pour_detected": self._poured,
            },
        )

    def is_task_success(self, **kwargs):
        return self._success
