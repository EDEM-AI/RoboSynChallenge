# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Validate clearance between the two table-rearrangement lanes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytorch_kinematics as pk
import torch

from embodichain.lab.sim.atomic_skills import JointPositionTarget
from embodichain.lab.sim.atomic_skills.runtime_commands import JointPositionPayload
from embodichain.lab.sim.cfg.source import _asset_file_path
from embodichain.lab.task_program.runtime.parallel_executor import ParallelSafetyError

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_skills import AtomicSkillEngine
    from embodichain.lab.sim.atomic_skills.runtime_commands import RuntimeCommandFrame
    from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc, RigidObject
    from embodichain.lab.sim.objects.robot import Robot
    from embodichain.lab.task_program.semantics.scene import SceneRegistry

__all__ = ["CutleryParallelSafetyFactory"]

_MAX_JOINT_STEP = 0.02
_MAX_INTERPOLATION_INTERVALS = 64
_CLEARANCE = 0.005


def _bounds(
    pose: torch.Tensor, center: torch.Tensor, half_extents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = pose[..., :3, :3]
    center = (rotation @ center[..., None]).squeeze(-1) + pose[..., :3, 3]
    radius = (rotation.abs() @ half_extents[..., None]).squeeze(-1)
    return center - radius, center + radius


def _lanes_are_clear(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> bool:
    left_low, left_high = left
    right_low, right_high = right
    gap = torch.maximum(
        left_low[:, :, None] - right_high[:, None, :],
        right_low[:, None, :] - left_high[:, :, None],
    )
    return bool((gap > _CLEARANCE).any(dim=-1).all())


def _local_box(shapes: Sequence[CollisionShapeDesc]) -> torch.Tensor:
    lows, highs = [], []
    for shape in shapes:
        if shape.vertices is not None:
            low, high = shape.vertices.amin(dim=0), shape.vertices.amax(dim=0)
        elif shape.half_extents is not None:
            low, high = -shape.half_extents, shape.half_extents
        else:
            raise ParallelSafetyError(
                f"Unsupported cutlery collision shape {shape.name!r}."
            )
        low, high = _bounds(shape.local_pose, (low + high) / 2, (high - low) / 2)
        lows.append(low)
        highs.append(high)
    low = torch.stack(lows).amin(dim=0)
    high = torch.stack(highs).amax(dim=0)
    return torch.stack(((low + high) / 2, (high - low) / 2))


@dataclass(frozen=True)
class _Lane:
    link_names: tuple[str, ...]
    link_boxes: torch.Tensor
    gripper_link: str
    utensil: RigidObject
    utensil_box: torch.Tensor

    def bounds(
        self, poses: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        links = torch.stack([poses[name] for name in self.link_names], dim=1)
        low, high = _bounds(links, self.link_boxes[:, 0], self.link_boxes[:, 1])
        gripper = poses[self.gripper_link]
        observed = self.utensil.get_local_pose(to_matrix=True).to(gripper)
        # Enclose the utensil both at its measured pose and under motion of
        # its gripper. This covers pickup and release without guessing contact.
        transported = gripper @ torch.linalg.solve(gripper[:1], observed)
        observed_low, observed_high = _bounds(observed, *self.utensil_box)
        move_low, move_high = _bounds(transported, *self.utensil_box)
        return (
            torch.cat((low, torch.minimum(observed_low, move_low)[:, None]), dim=1),
            torch.cat((high, torch.maximum(observed_high, move_high)[:, None]), dim=1),
        )


def _lane(robot: Robot, utensil: RigidObject, side: str) -> _Lane:
    link_boxes = {}
    for name in robot.link_names:
        if name.startswith(f"{side}_"):
            shapes = robot.get_link_collision_shapes(name)
            if shapes:
                link_boxes[name] = _local_box(shapes)
    gripper = f"{side}_gripper_base"
    fingers = {f"{side}_link7", f"{side}_link8"}
    shapes = utensil.get_collision_shapes()
    if gripper not in link_boxes or not fingers.issubset(link_boxes) or not shapes:
        raise ParallelSafetyError(
            "Cutlery lanes require native gripper and utensil geometry."
        )
    return _Lane(
        tuple(link_boxes),
        torch.stack(tuple(link_boxes.values())).to(
            device=robot.device, dtype=torch.float64
        ),
        gripper,
        utensil,
        _local_box(shapes).to(device=robot.device, dtype=torch.float64),
    )


class _CutleryParallelSafety:
    def __init__(self, robot: Robot, fork: RigidObject, spoon: RigidObject) -> None:
        self._robot = robot
        self._left = _lane(robot, fork, "left")
        self._right = _lane(robot, spoon, "right")
        asset = _asset_file_path(robot.cfg)
        if asset is None or Path(asset).suffix.lower() != ".urdf":
            raise ParallelSafetyError(
                "Cutlery clearance requires the assembled robot URDF."
            )
        self._chain = pk.build_chain_from_urdf(Path(asset).read_bytes()).to(
            device=robot.device, dtype=torch.float64
        )
        self._joint_order = [
            robot.joint_names.index(name)
            for name in self._chain.get_joint_parameter_names()
        ]
        names = tuple(dict.fromkeys((*self._left.link_names, *self._right.link_names)))
        self._frames = self._chain.get_frame_indices(*names)

    def validate(
        self,
        *,
        branch_frames: Mapping[str, RuntimeCommandFrame],
        merged_frame: RuntimeCommandFrame,
    ) -> None:
        if merged_frame.env_ids.numel() != 1 or int(merged_frame.env_ids[0]) != 0:
            raise ParallelSafetyError("Cutlery clearance is bound to one native arena.")
        if not bool(merged_frame.active_mask[0]):
            return
        current = self._robot.get_qpos(target=False).to(dtype=torch.float64)
        # A finished lane keeps its last commanded position while the peer moves.
        target = self._robot.get_qpos(target=True).to(current).clone()
        for command in merged_frame.commands:
            if (
                type(command.target) is not JointPositionTarget
                or type(command.payload) is not JointPositionPayload
            ):
                raise ParallelSafetyError(
                    "Cutlery lanes require joint-position commands."
                )
            target[:, list(command.target.joint_ids)] = command.payload.positions.to(
                target
            )
        target = self._robot._project_mimic_qpos(target)
        delta = target - current
        if not bool(torch.isfinite(delta).all()):
            raise ParallelSafetyError("Cutlery joint targets must be finite.")
        intervals = max(1, math.ceil(float(delta.abs().max()) / _MAX_JOINT_STEP))
        if intervals > _MAX_INTERPOLATION_INTERVALS:
            raise ParallelSafetyError(
                "Cutlery joint step exceeds the clearance sampling budget."
            )
        fractions = torch.linspace(
            0, 1, intervals + 1, device=current.device, dtype=current.dtype
        )
        samples = current + fractions[:, None] * delta
        root = self._robot.get_local_pose(to_matrix=True).to(current)
        poses = {
            name: root @ transform.get_matrix()
            for name, transform in self._chain.forward_kinematics(
                samples[:, self._joint_order], frame_indices=self._frames
            ).items()
        }
        if not _lanes_are_clear(self._left.bounds(poses), self._right.bounds(poses)):
            raise ParallelSafetyError(
                "Simultaneous cutlery motions lack cross-lane clearance."
            )


@dataclass(frozen=True, slots=True)
class CutleryParallelSafetyFactory:
    """Build a sampled cross-lane collision guard from cooked physical geometry."""

    validator_id: ClassVar[str] = "robosyn.table_rearrangement.cutlery_clearance"
    revision: ClassVar[str] = "1"
    supported_transport_ids: ClassVar[frozenset[str]] = frozenset(
        {JointPositionTarget.TRANSPORT_ID}
    )

    def create(
        self,
        *,
        simulation: SimulationManager,
        robot: Robot,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> _CutleryParallelSafety:
        """Bind one validator to the two native cutlery objects and the robot."""
        return _CutleryParallelSafety(
            robot,
            simulation.get_rigid_object("fork"),
            simulation.get_rigid_object("spoon"),
        )
