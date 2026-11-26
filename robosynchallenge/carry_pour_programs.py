"""Measure physical task outcomes for the native carry and pour programs."""

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
from .task_program_adapter import compose_task_program

_POUR_ANGLE = -1.10
_BOTTLE_MOUTH_HEIGHT = 0.236


class BottlePickup(RegisteredSemanticLowerer):
    """Approach upright bottles at the body and fallen bottles from above."""

    call_id = "robosyn.pick_bottle"
    revision = "1"
    target_descriptor = PickUp.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH
    requires_unoccupied_slots = ("primary",)

    def __init__(self, semantics, right_base):
        self.semantics = semantics
        self.right_base = right_base

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
        if not bool((axis[:, 2] > 0.90).all()):
            return SemanticLowering(
                goal=GraspGoal(semantics=self.semantics), registered_effect=effect
            )
        center = bottle[:, :3, 3] + 0.105 * axis
        direction = center - self.right_base[context.env_ids].to(center)
        direction[:, 2] = 0.0
        direction = torch.nn.functional.normalize(direction, dim=1)
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


class BottlePickupFactory:
    call_id = BottlePickup.call_id
    revision = BottlePickup.revision
    target_descriptor = BottlePickup.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        bottle = scene_registry.resolve("bottle", expected_type=SceneObjectRef)
        affordance = scene_registry.resolve_affordance(
            bottle, capability="affordance.grasp", explicit="bottle_grasp"
        )
        return BottlePickup(
            scene_registry.object_semantics(bottle, affordance=affordance),
            robot.get_link_pose("right_arm_base")[:, :3].clone(),
        )


class BottlePourPosition(RegisteredSemanticLowerer):
    """Find an upright bottle pose whose in-place tilt reaches the cup."""

    call_id = "robosyn.position_bottle_to_pour"
    revision = "1"
    target_descriptor = MoveHeldObject.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH

    def __init__(self, robot):
        self.robot = robot

    def lower(self, call, *, context, bound, option_template):
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        arm = endpoint.runtime_target
        held = context.task.get_held_object(endpoint.task_state_key)
        if held is None or held.semantics.entity_id != "bottle":
            raise ValueError("Pour positioning requires the held bottle.")
        cup = context.scene.entities["cup"].pose
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        headings = cup.new_tensor(
            (-1.05, -0.78, -0.52, -0.26, 0, 0.26, 0.52, 0.78, 1.05)
        )
        heights = cup.new_tensor((0.035, 0.06, 0.085, 0.11))
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
        pour_targets = targets.clone()
        pour_targets[..., :3, :3] = tilted
        object_to_tcp = held.object_to_eef.to(cup)
        if object_to_tcp.ndim == 2:
            object_to_tcp = object_to_tcp.expand(context.batch_size, -1, -1)
        poses = (
            torch.stack((targets, pour_targets), dim=2) @ object_to_tcp[:, None, None]
        ).flatten(1, 2)
        seeds = joints[:, None].expand(-1, poses.shape[1], -1)
        valid, qpos = self.robot.compute_batch_ik(
            pose=poses, joint_seed=seeds, name=arm.control_part, env_ids=context.env_ids
        )
        valid = valid.reshape(context.batch_size, len(yaw), 2).all(dim=-1)
        cost = (
            (qpos - seeds)
            .square()
            .sum(dim=-1)
            .reshape(context.batch_size, len(yaw), 2)
            .sum(dim=-1)
            .masked_fill(~valid, torch.inf)
        )
        rows = torch.arange(context.batch_size, device=cup.device)
        selected = targets[rows, cost.argmin(dim=1)]
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


class BottlePourPositionFactory:
    call_id = BottlePourPosition.call_id
    revision = BottlePourPosition.revision
    target_descriptor = BottlePourPosition.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return BottlePourPosition(robot)


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
            yaw_candidates=(0.0, 0.18, -0.18, math.pi, math.pi + 0.18, math.pi - 0.18),
        ),
        BasketCarryFactory(),
    ),
    additional_lowerers=(BasketHandlePickupFactory(),),
)
_BASKET_PROGRAM = load_task_program(
    _BASKET_DEPLOYMENT.program_path,
    integration=_BASKET_DEPLOYMENT.selection,
    validation_context=_BASKET_ADAPTER.registration.catalog,
)
_POUR_DEPLOYMENT, _POUR_ADAPTER = compose_task_program(
    "water_pouring",
    (),
    additional_lowerers=(BottlePickupFactory(), BottlePourPositionFactory()),
)
_POUR_PROGRAM = load_task_program(
    _POUR_DEPLOYMENT.program_path,
    integration=_POUR_DEPLOYMENT.selection,
    validation_context=_POUR_ADAPTER.registration.catalog,
)


@register_env(
    "RoboSyn-HandleBasket-v1",
    max_episode_steps=1500,
    task_program_registration=_BASKET_ADAPTER.registration,
    task_program_adapter_factory=_BASKET_ADAPTER,
)
class HandleBasket(EmbodiedEnv):
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
