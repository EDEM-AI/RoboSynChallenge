"""Qualify the Task Program's two-arm docking with measured tube contact."""

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

from .insertion_targets import cylinder_grasp_frame
from .task_program_adapter import compose_task_program


def _rank_joint_paths(robot, context, arm, joints, waypoints):
    batch, candidates, count = waypoints.shape[:3]
    poses = waypoints.flatten(1, 2)
    seeds = joints[:, None].expand(-1, poses.shape[1], -1)
    valid, solutions = robot.compute_batch_ik(
        pose=poses, joint_seed=seeds, name=arm.control_part, env_ids=context.env_ids
    )
    valid = valid.reshape(batch, candidates, count).all(dim=-1)
    solutions = solutions.reshape(batch, candidates, count, -1)
    start_cost = (solutions[:, :, 0] - joints[:, None]).square().sum(dim=-1)
    motion_cost = solutions.diff(dim=2).square().sum(dim=(-1, -2))
    return (start_cost + motion_cost).masked_fill(~valid, torch.inf), valid.any(dim=1)


class TubePickup(RegisteredSemanticLowerer):
    """Screen both jaw orientations at the middle of the tube."""

    call_id = "robosyn.pick_tube"
    revision = "1"
    target_descriptor = PickUp.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH
    requires_unoccupied_slots = ("primary",)

    def __init__(self, robot, semantics):
        self.robot = robot
        self.semantics = semantics

    def lower(self, call, *, context, bound, option_template):
        object_id = call.arguments["object"]
        tube = context.scene.entities[object_id].pose
        slants = tube.new_tensor(
            ((0, 0, 0), (0.14, 0, 0), (-0.14, 0, 0), (0, 0.22, 0), (0, -0.22, 0))
        )
        rotations = axis_angle_to_rotation_matrix(slants)
        flip = tube.new_tensor(((-1, 0, 0), (0, -1, 0), (0, 0, 1)))
        rotations = torch.cat((rotations, rotations @ flip), dim=0)
        top_down = cylinder_grasp_frame(tube[:, :3, 0])
        grasps = tube[:, None].expand(-1, len(rotations), -1, -1).clone()
        grasps[..., :3, :3] = top_down[:, None] @ rotations
        pre_grasp = grasps.clone()
        pre_grasp[..., :3, 3] -= grasps[..., :3, 2] * option_template.pre_grasp_distance
        half_lift = grasps.clone()
        half_lift[..., 2, 3] += option_template.lift_height / 2
        lift = grasps.clone()
        lift[..., 2, 3] += option_template.lift_height
        arm = bound.binding.resources["primary"].endpoints["motion"].runtime_target
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        cost, feasible = _rank_joint_paths(
            self.robot, context, arm, joints,
            torch.stack((pre_grasp, grasps, half_lift, lift), dim=2),
        )
        rows = torch.arange(context.batch_size, device=tube.device)
        return SemanticLowering(
            goal=GraspGoal(
                semantics=self.semantics[object_id],
                grasp_xpos=grasps[rows, cost.argmin(dim=1)],
            ),
            failed_preconditions={
                int(context.env_ids[row]): f"No reachable grasp at the middle of the {object_id}."
                for row in (~feasible).nonzero().flatten().tolist()
            },
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=object_id,
                        slot_id="primary",
                    ),
                ),
            ),
        )


class TubePickupFactory:
    call_id = TubePickup.call_id
    revision = TubePickup.revision
    target_descriptor = TubePickup.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        semantics = {}
        for object_id in ("sleeve", "insert"):
            entity = scene_registry.resolve(object_id, expected_type=SceneObjectRef)
            affordance = scene_registry.resolve_affordance(
                entity, capability="affordance.grasp", explicit=f"{object_id}_grasp"
            )
            semantics[object_id] = scene_registry.object_semantics(
                entity, affordance=affordance
            )
        return TubePickup(robot, semantics)


def _tube_contact_distance(sensor, sleeve_id, insert_id):
    contacts = sensor.get_data()
    a, b = contacts["user_ids"].unbind(dim=-1)
    pair = ((a == sleeve_id) & (b == insert_id)) | (
        (a == insert_id) & (b == sleeve_id)
    )
    return contacts["distance"].masked_fill(
        ~(pair & contacts["is_valid"]), torch.inf
    ).min(dim=1).values


class TubeTransport(RegisteredSemanticLowerer):
    """Keep the tube axis fixed while choosing a reachable wrist roll."""

    call_id = "robosyn.move_tube"
    revision = "1"
    target_descriptor = MoveHeldObject.descriptor()
    effect_contract_kind = SemanticEffectKind.ATTACH

    def __init__(self, robot, tube_end_distance):
        self.robot = robot
        self.tube_end_distance = tube_end_distance

    def lower(self, call, *, context, bound, option_template):
        object_id = call.arguments["object"]
        phase = call.arguments["phase"]
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        arm = endpoint.runtime_target
        held = context.task.get_held_object(endpoint.task_state_key)
        if held is None or held.semantics.entity_id != object_id:
            raise ValueError(f"Tube transport requires the held {object_id}.")

        sleeve = context.scene.entities["sleeve"].pose
        rolls = torch.arange(16, device=sleeve.device, dtype=sleeve.dtype)
        rolls *= math.pi / 8
        rotations = axis_angle_to_rotation_matrix(
            rolls[:, None] * sleeve.new_tensor((1, 0, 0))
        )
        measured = context.scene.entities[object_id].pose
        if object_id == "sleeve":
            target = sleeve.clone()
            target[:, :3, 3] = sleeve.new_tensor((0.49, -0.08, 0.97))
            target[:, :3, :3] = axis_angle_to_rotation_matrix(
                sleeve.new_tensor((0, 0, math.pi / 2))
            )
            targets = target[:, None, None].expand(-1, len(rolls), 1, -1, -1).clone()
            targets[..., :3, :3] = (
                target[:, None, None, :3, :3] @ rotations[None, :, None]
            )
        else:
            # Each deployment scales its meshes independently.
            end_distance = self.tube_end_distance[context.env_ids].to(sleeve)
            offsets = end_distance[:, None] + sleeve.new_tensor((0.024, 0.024))
            opposite = axis_angle_to_rotation_matrix(
                sleeve.new_tensor((0.0, 0.5, math.sqrt(3) / 2)) * math.pi
            )
            targets = (
                sleeve[:, None, None].expand(-1, len(rolls), offsets.shape[1], -1, -1).clone()
            )
            targets[..., :3, :3] = (
                sleeve[:, None, None, :3, :3] @ opposite @ rotations[None, :, None]
            )
            targets[..., :3, 3] = (
                sleeve[:, None, None, :3, 3]
                + offsets[:, None, :, None] * sleeve[:, None, None, :3, 0]
            )
            targets[:, :, 0, 2, 3] += 0.050
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        tcp = self.robot.compute_fk(
            qpos=joints, name=arm.control_part, to_matrix=True, env_ids=context.env_ids
        )
        measured_grasp = torch.linalg.solve(measured, tcp)
        wrists = targets @ measured_grasp[:, None, None]
        cost, feasible = _rank_joint_paths(self.robot, context, arm, joints, wrists)
        # Keep the bulk of both parallel grippers above the tubes so their
        # housings cannot meet ahead of the tube ends.
        cost = cost + 2 * (1 + wrists[:, :, 0, 2, 2])
        rows = torch.arange(context.batch_size, device=sleeve.device)
        selected = targets[rows, cost.argmin(dim=1)]
        # MoveHeldObject uses the verified attachment as its planning frame.
        verified_grasp = held.object_to_eef.to(sleeve).expand_as(measured_grasp)
        frame_correction = measured_grasp @ torch.linalg.inv(verified_grasp)
        return SemanticLowering(
            goal=HeldObjectPoseGoal(
                selected @ frame_correction[:, None], object_id=object_id
            ),
            failed_preconditions={
                int(context.env_ids[row]): f"No reachable roll for the {phase} pose."
                for row in (~feasible).nonzero().flatten().tolist()
            },
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=object_id,
                        slot_id="primary",
                        retain_during_segments=("transport",),
                    ),
                ),
            ),
        )


class TubeTransportFactory:
    call_id = TubeTransport.call_id
    revision = TubeTransport.revision
    target_descriptor = TubeTransport.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return TubeTransport(robot, _tube_end_distance(simulation))


def _tube_end_distance(simulation):
    sleeve = simulation.get_rigid_object("guijiao1").get_vertices(scale=True)
    insert = simulation.get_rigid_object("guijiao2").get_vertices(scale=True)
    return sleeve[..., 0].amax(dim=-1) + insert[..., 0].amax(dim=-1)


class TubeDock(RegisteredSemanticLowerer):
    """Keep both jaws loaded while advancing along the live sleeve axis."""

    call_id = "robosyn.dock_tubes"
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state = True

    def __init__(self, robot, tube_end_distance, contact_sensor):
        self.robot = robot
        self.tube_end_distance = tube_end_distance
        self.contact_sensor = contact_sensor
        self.sleeve_id = contact_sensor.get_actor_ids("guijiao1")
        self.insert_id = contact_sensor.get_actor_ids("guijiao2")

    def lower(self, call, *, context, bound, option_template):
        joint_ids = list(
            bound.binding.resources["primary"].endpoints["motion"].runtime_target.joint_ids
        )
        arm_ids = list(self.robot.get_joint_ids("left_arm"))
        arm = [joint_ids.index(index) for index in arm_ids]
        jaws = [
            joint_ids.index(index)
            for part in ("left_eef", "right_eef")
            for index in self.robot.get_joint_ids(part)
        ]
        start = context.robot.qpos[:, arm_ids]
        tcp = self.robot.compute_fk(
            qpos=start, name="left_arm", to_matrix=True, env_ids=context.env_ids
        )
        sleeve = context.scene.entities["sleeve"].pose
        insert = context.scene.entities["insert"].pose
        sleeve_axis = sleeve[:, :3, 0]
        end_distance = self.tube_end_distance[context.env_ids].to(sleeve)
        centers = insert[:, :3, 3] - sleeve[:, :3, 3]
        measured_gap = (centers * sleeve_axis).sum(dim=-1)
        if call.arguments["phase"] == "approach":
            desired_gap = end_distance + 0.008
        else:
            advance = torch.where(measured_gap > end_distance + 0.007, 0.002, 0.001)
            desired_gap = torch.maximum(measured_gap - advance, end_distance - 0.004)
        # The first physical contact is enough. Further travel tilts the tubes
        # out of alignment even if both jaw drives remain closed.
        touching = _tube_contact_distance(
            self.contact_sensor, self.sleeve_id, self.insert_id
        )[context.env_ids] <= 0.003
        target = tcp.clone()
        target[:, :3, 3] += (
            sleeve_axis * desired_gap[:, None] - centers
        )
        valid, stroke = self.robot.compute_batch_ik(
            pose=target[:, None],
            joint_seed=start[:, None],
            name="left_arm",
            env_ids=context.env_ids,
        )
        path = context.robot.qpos[:, None, joint_ids].expand(-1, 2, -1).clone()
        path[:, 1, arm] = stroke[:, 0]
        path[..., jaws] = 0.0
        last_target = self.robot.get_qpos(target=True)[context.env_ids][:, joint_ids]
        path = torch.where(touching[:, None, None], last_target[:, None], path)
        reachable = valid[:, 0].bool() | touching
        return SemanticLowering(
            goal=JointPositionGoal(path),
            failed_preconditions={
                int(context.env_ids[row]): "The left arm cannot reach the tube ends."
                for row in (~reachable).nonzero().flatten().tolist()
            },
        )


class TubeDockFactory:
    call_id = TubeDock.call_id
    revision = "1"
    target_descriptor = TubeDock.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return TubeDock(
            robot, _tube_end_distance(simulation), simulation.get_sensor("guijiao_contact")
        )


_DEPLOYMENT, _ADAPTER = compose_task_program(
    "item_assembly", (),
    additional_lowerers=(TubePickupFactory(), TubeTransportFactory(), TubeDockFactory()),
)
_PROGRAM = load_task_program(
    _DEPLOYMENT.program_path,
    integration=_DEPLOYMENT.selection,
    validation_context=_ADAPTER.registration.catalog,
)


@register_env(
    "RoboSyn-ItemAssembly-v1",
    max_episode_steps=1500,
    task_program_registration=_ADAPTER.registration,
    task_program_adapter_factory=_ADAPTER,
)
class ItemAssembly(EmbodiedEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.task_program is None:
            cfg.task_program = _PROGRAM
        super().__init__(cfg=cfg, **kwargs)
        self._sleeve = self.sim.get_rigid_object("guijiao1")
        self._insert = self.sim.get_rigid_object("guijiao2")
        self._contact_sensor = self.sim.get_sensor("guijiao_contact")
        self._sleeve_id = self._contact_sensor.get_actor_ids("guijiao1")
        self._insert_id = self._contact_sensor.get_actor_ids("guijiao2")

    def _docking_metrics(self):
        sleeve = self._sleeve.get_local_pose(to_matrix=True)
        insert = self._insert.get_local_pose(to_matrix=True)
        sleeve_axis = sleeve[:, :3, 0]
        alignment = (sleeve_axis * insert[:, :3, 0]).sum(dim=-1).abs().clamp_max(1)
        angle = torch.acos(alignment)
        delta = insert[:, :3, 3] - sleeve[:, :3, 3]
        along_axis = (delta * sleeve_axis).sum(dim=-1, keepdim=True) * sleeve_axis
        lateral = torch.linalg.vector_norm(delta - along_axis, dim=-1)

        minimum = _tube_contact_distance(
            self._contact_sensor, self._sleeve_id, self._insert_id
        )
        return angle, lateral, minimum

    def compute_task_state(self, **kwargs):
        angle, lateral, separation = self._docking_metrics()
        keep_running = torch.zeros_like(angle, dtype=torch.bool)
        return (
            keep_running,
            keep_running,
            {
                "tube_angle_rad": angle,
                "tube_lateral_offset_m": lateral,
                "tube_contact_separation_m": separation,
            },
        )

    def is_task_success(self, **kwargs):
        angle, lateral, separation = self._docking_metrics()
        return (
            (angle <= math.radians(15))
            & (lateral <= 0.02)
            & (separation <= 0.003)
            & (self._elapsed_steps >= 5)
        )
