"""Native deployments for the four additional RoboSynChallenge tasks."""

from dataclasses import dataclass, replace
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

from .carry_pour_programs import (
    BottlePickupFactory,
    BottlePourPositionFactory,
    WaterPouring,
)
from .insertion_targets import (
    InsertionAlignmentFactory,
    InsertionLoweringFactory,
    InsertionPlace,
    InsertionPlaceFactory,
)
from .pipette_program import (
    ApproachPipetteFactory,
    ManipulatePipette,
    PipetteGraspFactory,
    PipettePressFactory,
    PositionPipette,
    PositionPipetteFactory,
)
from .placement_programs import SampleLoading
from .task_program_adapter import compose_task_program


def _program(name, replacements=(), extensions=(), *, use_scene_grasp_options=False):
    deployment, adapter = compose_task_program(
        f"other_tasks/{name}",
        replacements,
        additional_lowerers=extensions,
        use_scene_grasp_options=use_scene_grasp_options,
    )
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=adapter.registration.catalog,
    )
    return deployment, adapter, program


class LidPickup(RegisteredSemanticLowerer):
    call_id = "robosyn.pick_lid"
    revision = "1"
    target_descriptor = PickUp.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH
    requires_unoccupied_slots = ("primary",)

    def __init__(self, semantics, robot):
        self.semantics = semantics
        self.robot = robot

    def lower(self, call, *, context, bound, option_template):
        lid = context.scene.entities["lid"].pose
        # Start from the authored knob grasp and screen its symmetric alternatives.
        knob = lid.new_tensor(
            ((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, -1, 0.032), (0, 0, 0, 1))
        )
        twists = lid.new_tensor((20, 30, 40, -20, -30, -40, 0)) * (math.pi / 180)
        tilts = axis_angle_to_rotation_matrix(
            twists[:, None] * lid.new_tensor((0, 1, 0))
        )
        flip = lid.new_tensor(((-1, 0, 0), (0, -1, 0), (0, 0, 1)))
        rotations = torch.cat((tilts, tilts @ flip), dim=0)
        poses = (lid @ knob)[:, None].expand(-1, len(rotations), -1, -1).clone()
        poses[..., :3, :3] = poses[..., :3, :3] @ rotations
        above = poses.clone()
        above[..., 2, 3] += option_template.pre_grasp_distance
        lift = poses.clone()
        lift[..., 2, 3] += option_template.lift_height
        waypoints = torch.stack((above, poses, lift), dim=2).flatten(1, 2)
        qpos = context.robot.qpos[:, self.robot.get_joint_ids("left_arm")]
        seeds = qpos[:, None].expand(-1, waypoints.shape[1], -1)
        valid, joints = self.robot.compute_batch_ik(
            pose=waypoints, joint_seed=seeds, name="left_arm", env_ids=context.env_ids
        )
        valid = valid.reshape(context.batch_size, len(rotations), 3).all(dim=-1)
        costs = (
            (joints - seeds)
            .square()
            .sum(dim=-1)
            .reshape(valid.shape + (3,))
            .sum(dim=-1)
        )
        best = costs.masked_fill(~valid, torch.inf).argmin(dim=-1)
        rows = torch.arange(context.batch_size, device=lid.device)
        grasp = GraspGoal(semantics=self.semantics, grasp_xpos=poses[rows, best])
        failures = {
            int(context.env_ids[row]): "No reachable pose for the lid knob."
            for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
        }
        return SemanticLowering(
            goal=grasp,
            failed_preconditions=failures,
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id="lid",
                        slot_id="primary",
                    ),
                ),
            ),
        )


class LidPickupFactory:
    call_id = LidPickup.call_id
    revision = LidPickup.revision
    target_descriptor = LidPickup.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        lid = scene_registry.resolve("lid", expected_type=SceneObjectRef)
        affordance = scene_registry.resolve_affordance(
            lid, capability="affordance.grasp", explicit="lid_grasp"
        )
        return LidPickup(
            scene_registry.object_semantics(lid, affordance=affordance), robot
        )


class LidClearance(RegisteredSemanticLowerer):
    """Move the lid to a reachable clearance while keeping its grasp."""

    call_id = "robosyn.move_lid_clear"
    revision = "1"
    target_descriptor = MoveHeldObject.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot):
        self.robot = robot

    def lower(self, call, *, context, bound, option_template):
        pan = context.scene.entities["pan"].pose
        lid = context.scene.entities["lid"].pose
        motion = bound.binding.resources["primary"].endpoints["motion"]
        arm = motion.runtime_target
        held = context.task.get_held_object(motion.task_state_key)
        if held is None or held.semantics.entity_id != "lid":
            raise ValueError("Lid clearance requires the held lid.")
        # Clear the rim plus the receiving arm's wrist envelope on the left.
        # The lid is still held while the right arm crosses to the pan.
        offsets = pan.new_tensor(
            (
                (-0.09, 0.225, 0.195),
                (-0.14, 0.215, 0.195),
                (-0.18, 0.195, 0.19),
                (-0.05, 0.235, 0.19),
                (-0.13, 0.225, 0.215),
                (-0.19, 0.165, 0.22),
            )
        )
        targets = lid[:, None].expand(-1, len(offsets), -1, -1).clone()
        targets[..., :3, 3] = pan[:, None, :3, 3] + offsets @ pan[:, :3, :3].mT
        grasp = held.object_to_eef.to(pan).expand_as(lid)
        wrists = targets @ grasp[:, None]
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        seeds = joints[:, None].expand(-1, len(offsets), -1)
        valid, solutions = self.robot.compute_batch_ik(
            pose=wrists, joint_seed=seeds, name=arm.control_part, env_ids=context.env_ids
        )
        valid = valid.to(dtype=torch.bool)
        cost = (solutions - seeds).square().sum(dim=-1).masked_fill(~valid, torch.inf)
        rows = torch.arange(context.batch_size, device=pan.device)
        return SemanticLowering(
            goal=HeldObjectPoseGoal(
                targets[rows, cost.argmin(dim=1)], world_yaw_free=True
            )
        )


class LidClearanceFactory:
    call_id = LidClearance.call_id
    revision = LidClearance.revision
    target_descriptor = LidClearance.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return LidClearance(robot)


class LidSeat(RegisteredSemanticLowerer):
    """Return the held lid above the rim before lowering it vertically."""

    call_id = "robosyn.seat_lid"
    revision = "1"
    target_descriptor = MoveHeldObject.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot):
        self.robot = robot

    def lower(self, call, *, context, bound, option_template):
        pan = context.scene.entities["pan"].pose
        lid = context.scene.entities["lid"].pose
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        arm = endpoint.runtime_target
        held = context.task.get_held_object(endpoint.task_state_key)
        if held is None or held.semantics.entity_id != "lid":
            raise ValueError("Seating the lid requires its held knob.")

        yaw = torch.atan2(lid[:, 1, 0], lid[:, 0, 0])
        turns = lid.new_tensor((0, -0.39, 0.39, -0.79, 0.79, -1.57, 1.57, math.pi))
        flat = axis_angle_to_rotation_matrix(
            (yaw[:, None, None] + turns[None, :, None]) * lid.new_tensor((0, 0, 1))
        )
        seat = lid[:, None].expand(-1, len(turns), -1, -1).clone()
        seat[..., :3, :3] = flat
        seat[..., :3, 3] = pan[:, None, :3, 3] + lid.new_tensor((0, 0, 0.092))
        above = seat.clone()
        above[..., 2, 3] += 0.055
        middle = above.clone()
        middle[..., :2, 3] = (pan[:, None, :2, 3] + lid[:, None, :2, 3]) / 2
        middle[..., 2, 3] += 0.020
        path = torch.stack((middle, above, seat), dim=2)

        joints = context.robot.qpos[:, list(arm.joint_ids)]
        wrist = self.robot.compute_fk(
            qpos=joints, name=arm.control_part, to_matrix=True, env_ids=context.env_ids
        )
        measured_grasp = torch.linalg.solve(lid, wrist)
        planned_grasp = held.object_to_eef.to(lid).expand_as(lid)
        wrists = (path @ measured_grasp[:, None, None]).flatten(1, 2)
        seeds = joints[:, None].expand(-1, wrists.shape[1], -1)
        valid, solutions = self.robot.compute_batch_ik(
            pose=wrists, joint_seed=seeds, name=arm.control_part, env_ids=context.env_ids
        )
        valid = valid.reshape(context.batch_size, len(turns), 3).all(dim=-1)
        cost = (solutions - seeds).square().sum(dim=-1).reshape(*valid.shape, 3)
        cost = (cost.sum(dim=-1) + 1 - turns.cos()).masked_fill(~valid, torch.inf)
        rows = torch.arange(context.batch_size, device=lid.device)
        selected = path[rows, cost.argmin(dim=1)]
        return SemanticLowering(
            goal=HeldObjectPoseGoal(
                selected @ (measured_grasp @ torch.linalg.inv(planned_grasp))[:, None]
            ),
            failed_preconditions={
                int(context.env_ids[row]): "There is no reachable path above the pan rim."
                for row in (~valid.any(dim=1)).nonzero().flatten().tolist()
            },
        )


class LidSeatFactory:
    call_id = LidSeat.call_id
    revision = LidSeat.revision
    target_descriptor = LidSeat.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return LidSeat(robot)


_CARROT_PLACEMENT = InsertionPlaceFactory(
    "carrot",
    "pan",
    (0, -0.035, 0.096),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    above_opening=0.018,
    arm_side="right",
    # Keep the long axis across the pan at the near side of its opening.
    yaw_candidates=(0.0, -0.22, 0.22, math.pi, math.pi - 0.22, math.pi + 0.22),
    tilt_candidates=(0.0, -0.26, 0.26),
)


class PanPlacement(InsertionPlace):
    def __init__(self, robot, *, align_lid_before_release):
        super().__init__(robot, None)
        upright = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        lid = InsertionPlaceFactory(
            "lid",
            "pan",
            (0, 0, 0.086),
            upright,
            above_opening=0.025,
            yaw_candidates=(-math.pi / 2, math.pi / 2, 0, math.pi),
        )
        if align_lid_before_release:
            lid = replace(lid, release_at_current_pose=True)
        self.placements = {
            "carrot": InsertionPlace(
                robot,
                replace(_CARROT_PLACEMENT, release_at_current_pose=True),
            ),
            "lid": InsertionPlace(robot, lid),
        }

    def lower(self, call, *, context, bound, option_template):
        return self.placements[call.arguments["object"]].lower(
            call, context=context, bound=bound, option_template=option_template
        )


@dataclass(frozen=True)
class PanPlacementFactory:
    align_lid_before_release: bool = False

    call_id = InsertionPlaceFactory.call_id
    revision = InsertionPlaceFactory.revision
    target_descriptor = InsertionPlaceFactory.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return PanPlacement(robot, align_lid_before_release=self.align_lid_before_release)


_PAN_LOWERERS = (
    LidPickupFactory(),
    LidClearanceFactory(),
    LidSeatFactory(),
    InsertionAlignmentFactory(
        _CARROT_PLACEMENT, lift_height=0.018, include_release_pose=True
    ),
)
_, _PAN_ADAPTER, _PAN_PROGRAM_WITH_ALIGNMENT = _program(
    "open_pan", (PanPlacementFactory(),), _PAN_LOWERERS
)
_, _PAN_ALIGNED_ADAPTER, _PAN_ALIGNED_PROGRAM = _program(
    "open_pan", (PanPlacementFactory(align_lid_before_release=True),), _PAN_LOWERERS
)
_PAN_PROGRAM = replace(
    _PAN_PROGRAM_WITH_ALIGNMENT,
    program=replace(
        _PAN_PROGRAM_WITH_ALIGNMENT.program,
        items=tuple(
            segment
            for segment in _PAN_PROGRAM_WITH_ALIGNMENT.program.items
            if segment.name != "align_lid"
        ),
    ),
)


@register_env(
    "RoboSyn-OpenPan-v1",
    max_episode_steps=1900,
    task_program_registration=_PAN_ADAPTER.registration,
    task_program_adapter_factory=_PAN_ADAPTER,
)
class OpenPan(EmbodiedEnv):
    program = _PAN_PROGRAM

    def __init__(self, cfg, **kwargs):
        align_lid = (cfg.extensions or {}).get("align_lid_before_release", False)
        if cfg.task_program is None:
            cfg.task_program = _PAN_ALIGNED_PROGRAM if align_lid else self.program
        if align_lid:
            adapter = kwargs.get("task_program_adapter_factory")
            if adapter is None or adapter is _PAN_ADAPTER:
                kwargs["task_program_adapter_factory"] = _PAN_ALIGNED_ADAPTER
        super().__init__(cfg=cfg, **kwargs)
        self._pan = self.sim.get_rigid_object("pan")
        self._lid = self.sim.get_rigid_object("lid")
        self._carrot = self.sim.get_rigid_object("carrot")
        self._opened = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._opened[(options or {}).get("reset_ids", slice(None))] = False
        return obs, info

    def _measure(self):
        pan = self._pan.get_local_pose()
        lid = self._lid.get_local_pose(to_matrix=True)
        carrot = self._carrot.get_local_pose()
        lid_delta = lid[:, :3, 3] - pan[:, :3]
        carrot_delta = carrot[:, :3] - pan[:, :3]
        lid_distance = lid_delta[:, :2].norm(dim=1)
        carrot_distance = carrot_delta[:, :2].norm(dim=1)
        self._opened |= (lid_delta[:, 2] > 0.15) & (lid_distance > 0.12)
        carrot_inside = (
            (carrot_distance < 0.10)
            & (carrot_delta[:, 2] > -0.03)
            & (carrot_delta[:, 2] < 0.09)
        )
        lid_closed = (
            (lid_distance < 0.08)
            & (lid_delta[:, 2] > 0.04)
            & (lid_delta[:, 2] < 0.11)
            & (lid[:, 2, 2] > 0.90)
        )
        success = self._opened & carrot_inside & lid_closed
        return (
            success,
            {
                "pan_opened": self._opened,
                "lid_pan_distance_m": lid_distance,
                "carrot_pan_distance_m": carrot_distance,
                "lid_pan_height_m": lid_delta[:, 2],
                "carrot_pan_height_m": carrot_delta[:, 2],
            },
        )

    def compute_task_state(self, **kwargs):
        success, metrics = self._measure()
        keep_running = torch.zeros_like(success)
        return keep_running, keep_running, metrics

    def is_task_success(self, **kwargs):
        return self._measure()[0]


class PositionOverDestination(PositionPipette):
    call_id = "robosyn.position_pipette_at_destination"


class PositionOverDestinationFactory(PositionPipetteFactory):
    call_id = PositionOverDestination.call_id

    def create(self, *, simulation, robot, scene_registry, engine):
        return PositionOverDestination(robot, "beaker2", clear_beaker_rim=True)


class PositionOverSourceFactory(PositionPipetteFactory):
    clear_beaker_rim = True


_, _TRANSFER_ADAPTER, _TRANSFER_PROGRAM = _program(
    "manipulate_pipette_two_beaker",
    extensions=(
        ApproachPipetteFactory(),
        PipetteGraspFactory(),
        PositionOverSourceFactory(),
        PositionOverDestinationFactory(),
        PipettePressFactory(),
    ),
)


@register_env(
    "RoboSyn-ManipulatePipetteTwoBeaker-v1",
    max_episode_steps=1900,
    task_program_registration=_TRANSFER_ADAPTER.registration,
    task_program_adapter_factory=_TRANSFER_ADAPTER,
)
class TransferPipette(ManipulatePipette):
    program = _TRANSFER_PROGRAM

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self._destination = self.sim.get_rigid_object("beaker2")

    def compute_task_state(self, **kwargs):
        qpos = self._pipette.get_qpos()[:, 0]
        minimum = self._pipette.get_qpos_limits()[:, 0, 0]
        down = qpos <= 0.66 * minimum + 1e-4
        new_press = down & ~self._pressed
        expected_beaker = torch.where(
            self._press_count == 0,
            self._over_upright_beaker(),
            self._over_upright_beaker(self._destination),
        )
        self._press_count += (
            new_press & expected_beaker & (self._press_count < 2)
        ).long()
        self._pressed.copy_(down)
        running = torch.zeros_like(down)
        return running, running, {"pipette_press_count": self._press_count}

    def is_task_success(self, **kwargs):
        return (
            (self._press_count == 2)
            & ~self._pressed
            & self._over_upright_beaker(self._destination)
        )


_, _POUR_ADAPTER, _POUR_PROGRAM = _program(
    "pour_water", extensions=(BottlePickupFactory(), BottlePourPositionFactory())
)


@register_env(
    "RoboSyn-PourWater-v1",
    max_episode_steps=1500,
    task_program_registration=_POUR_ADAPTER.registration,
    task_program_adapter_factory=_POUR_ADAPTER,
)
class PourWater(WaterPouring):
    program = _POUR_PROGRAM


_DIRECT_INSERTION = InsertionPlaceFactory(
    "test_tube",
    "rack",
    (0.01105, -0.0001, 0.115),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    arm_side="right",
    follow_receptacle_vertical=True,
)
_TUBE_DEPLOYMENT, _TUBE_ADAPTER, _TUBE_PROGRAM = _program(
    "sample_loading",
    (replace(_DIRECT_INSERTION, release_at_current_pose=True),),
    (
        InsertionAlignmentFactory(_DIRECT_INSERTION),
        InsertionLoweringFactory(_DIRECT_INSERTION),
    ),
    use_scene_grasp_options=True,
)


@register_env(
    "RoboSyn-SingleArmSampleLoading-v1",
    max_episode_steps=1500,
    task_program_registration=_TUBE_ADAPTER.registration,
    task_program_adapter_factory=_TUBE_ADAPTER,
)
class SingleArmSampleLoading(SampleLoading):
    deployment = _TUBE_DEPLOYMENT
    adapter = _TUBE_ADAPTER
    program = _TUBE_PROGRAM
