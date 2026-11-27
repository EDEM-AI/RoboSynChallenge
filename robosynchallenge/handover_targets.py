"""Filter native handover grasps using the receiving arm's kinematics."""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from embodichain.lab.sim.atomic_skills import HandOver, HandOverGoal
from embodichain.lab.sim.atomic_skills.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_skills.affordance_sampling import AffordancePoseCandidates
from embodichain.lab.task_program.compiler.lowering import (
    HeldObjectRelation,
    RegisteredHeldObjectEffect,
    RegisteredSemanticEffect,
    RegisteredSemanticLowerer,
    SemanticEffectKind,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import SceneObjectRef


@dataclass
class ReachableAntipodalAffordance(AntipodalAffordance):
    """Keep native antipodal candidates that the receiving wrist can reach."""

    screen: Callable[
        [AffordancePoseCandidates, torch.Tensor], AffordancePoseCandidates
    ] | None = None

    def get_grasp_candidates(self, generator, object_poses, approach_direction, **region):
        candidates = super().get_grasp_candidates(
            generator, object_poses, approach_direction, **region
        )
        return candidates if self.screen is None else self.screen(candidates, object_poses)


class ReachableHandover(RegisteredSemanticLowerer):
    """Ground a native HandOver that checks the receiver's approach."""
    revision = "1"
    target_descriptor = HandOver.descriptor()
    effect_contract_kind = SemanticEffectKind.TRANSFER
    requires_unoccupied_slots = ("destination",)

    def __init__(self, robot, semantics):
        self.robot = robot
        self.semantics = semantics

    def lower(self, call, *, context, bound, option_template):
        arm = bound.binding.resources["destination"].endpoints["motion"].runtime_target
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        env_ids = torch.as_tensor(context.env_ids, device=joints.device)

        def screen(candidates, exchanged_pen):
            candidates = self.extend_candidates(candidates, exchanged_pen)
            poses = candidates.poses
            count = poses.shape[1]
            offsets = poses.new_tensor((1.0, 0.5, 0.0)) * option_template.pre_grasp_distance
            path = poses[:, :, None].expand(-1, -1, len(offsets), -1, -1).clone()
            path[..., :3, 3] -= poses[:, :, None, :3, 2] * offsets[None, None, :, None]
            seeds = joints.repeat_interleave(count, dim=0)
            valid, solutions = self.robot.compute_batch_ik(
                pose=path.flatten(0, 1),
                joint_seed=seeds,
                name=arm.control_part,
                env_ids=env_ids.repeat_interleave(count),
                continuous=True,
            )
            valid = valid.reshape(context.batch_size, count, len(offsets)).all(dim=-1)
            motion = (solutions[:, 0] - seeds).square().sum(dim=-1)
            motion += solutions.diff(dim=1).square().sum(dim=(-1, -2))
            valid = candidates.valid & valid
            cost = candidates.costs + 0.02 * motion.reshape(valid.shape)
            return self.rank_candidates(
                context, arm, candidates, exchanged_pen, valid, cost,
                solutions[:, -1].reshape(context.batch_size, count, -1),
            )

        base = self.semantics.affordance
        affordance = ReachableAntipodalAffordance(
            mesh_vertices=base.mesh_vertices,
            mesh_triangles=base.mesh_triangles,
            screen=screen,
        )
        object_id = self.semantics.entity_id
        target = context.scene.entities[object_id].pose.clone()
        target[:, 2, 3].clamp_(min=0.93)
        return SemanticLowering(
            goal=HandOverGoal(replace(self.semantics, affordance=affordance), target),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.TRANSFER,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="source",
                        relation=HeldObjectRelation.DETACHED,
                        object_id=object_id,
                        slot_id="source",
                    ),
                    RegisteredHeldObjectEffect(
                        expectation_id="destination",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=object_id,
                        slot_id="destination",
                    ),
                ),
            ),
        )

    def extend_candidates(self, candidates, object_pose):
        """Retain the native candidates unless the task supplies a fallback."""
        return candidates

    @abstractmethod
    def rank_candidates(self, context, arm, candidates, object_pose, valid, cost, joints):
        """Apply the task's placement geometry to reachable receiver grasps."""

    def rank_insertion_candidates(
        self, context, arm, candidates, object_pose, valid, cost, joints,
        *, eligible, placement, clearance, reach_margin_weight=0.0,
    ):
        """Prefer receiver grasps that also reach the live receptacle opening."""
        poses = candidates.poses
        shortlist = cost.masked_fill(~eligible, torch.inf).topk(
            min(24, poses.shape[1]), dim=1, largest=False
        ).indices
        rows = torch.arange(context.batch_size, device=object_pose.device)[:, None]
        grasp = torch.linalg.solve(object_pose[:, None], poses[rows, shortlist])
        receptacle = context.scene.entities[placement.receptacle_id].pose
        release, pitch = placement.release_targets(receptacle, object_pose)
        above = placement.lift_targets(release, placement.above_opening + clearance)
        wrist_targets = torch.stack((above, release), dim=2)[:, None] @ grasp[:, :, None, None]
        count = shortlist.shape[1]
        seeds = joints[rows, shortlist, None, None].expand(-1, -1, len(pitch), 2, -1)
        feasible, solutions = self.robot.compute_batch_ik(
            pose=wrist_targets.flatten(1, 3),
            joint_seed=seeds.flatten(1, 3),
            name=arm.control_part,
            env_ids=context.env_ids,
        )
        layout = (context.batch_size, count, len(pitch), 2)
        feasible = feasible.reshape(layout).all(dim=-1) & eligible[rows, shortlist, None]
        motion = (solutions - seeds.flatten(1, 3)).square().sum(dim=-1)
        motion = motion.reshape(layout).sum(dim=-1).masked_fill(~feasible, torch.inf)
        insertion_cost = motion.amin(dim=2)
        reachable = torch.zeros_like(valid).scatter(1, shortlist, feasible.any(dim=2))
        reach_margin = feasible.to(cost.dtype).mean(dim=2)
        ranked = cost.clone().scatter(
            1,
            shortlist,
            cost[rows, shortlist] + 0.02 * insertion_cost
            + reach_margin_weight * (1 - reach_margin),
        )
        return replace(
            candidates,
            valid=torch.where(reachable.any(dim=1, keepdim=True), reachable, valid),
            costs=torch.where(reachable.any(dim=1, keepdim=True), ranked, cost),
        )


class ReachableHandoverFactory:
    revision = ReachableHandover.revision
    target_descriptor = ReachableHandover.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        entity = scene_registry.resolve(self.object_id, expected_type=SceneObjectRef)
        grasp = scene_registry.resolve_affordance(
            entity, capability="affordance.grasp", explicit=f"{self.object_id}_grasp"
        )
        return self.lowerer_type(
            robot, scene_registry.object_semantics(entity, affordance=grasp)
        )
