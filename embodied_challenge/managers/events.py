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

from __future__ import annotations

import torch
import numpy as np
import os
import random
import json

from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple, Dict

from embodichain.lab.sim.objects import (
    Light,
    RigidObject,
    RigidObjectGroup,
    Articulation,
    Robot,
)
from embodichain.lab.sim.cfg import RigidObjectCfg, ArticulationCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.utils.module_utils import find_function_from_modules
from embodichain.utils.string import remove_regex_chars, resolve_matching_names
from embodichain.utils.file import get_all_files_in_directory
from embodichain.utils.math import (
    sample_uniform,
    pose_inv,
    xyz_quat_to_4x4_matrix,
    trans_matrix_to_xyz_quat,
)
from embodichain.utils import logger
from embodichain.data import get_data_path

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


def print_articulation_attrs(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
) -> None:
    """Print runtime physical attributes of a loaded articulation.

    This reads values from physics bodies after loading, not from configuration files.
    """

    asset = env.sim.get_asset(entity_cfg.uid)
    if asset is None:
        logger.log_error(
            f"Cannot print articulation attrs: asset '{entity_cfg.uid}' not found."
        )
        return

    if not isinstance(asset, Articulation):
        logger.log_warning(
            f"Asset '{entity_cfg.uid}' is type {type(asset)}, not Articulation. Skipping."
        )
        return

    runtime_attrs = {}
    getter_names = [
        "get_mass",
        "get_static_friction",
        "get_dynamic_friction",
        "get_restitution",
        "get_linear_damping",
        "get_angular_damping",
        "get_contact_offset",
        "get_rest_offset",
        "get_max_depenetration_velocity",
    ]

    def _to_python_scalar(val):
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return float(val.detach().cpu().item())
            return val.detach().cpu().tolist()
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return float(val.item())
            return val.tolist()
        if isinstance(val, (int, float, bool, str)):
            return val
        return str(val)

    try:
        first_entity = asset._entities[0]
        if hasattr(first_entity, "get_link_names") and hasattr(first_entity, "get_physical_body"):
            for link_name in first_entity.get_link_names():
                body = first_entity.get_physical_body(link_name)
                runtime_attrs[link_name] = {}
                for getter_name in getter_names:
                    getter = getattr(body, getter_name, None)
                    if callable(getter):
                        try:
                            runtime_attrs[link_name][getter_name] = _to_python_scalar(
                                getter()
                            )
                        except Exception as exc:
                            runtime_attrs[link_name][getter_name] = (
                                f"<error: {exc}>"
                            )

                # Fallback: if no known getter is available, inspect all no-arg get_* methods.
                if len(runtime_attrs[link_name]) == 0:
                    for attr_name in dir(body):
                        if not attr_name.startswith("get_"):
                            continue
                        getter = getattr(body, attr_name, None)
                        if callable(getter):
                            try:
                                runtime_attrs[link_name][attr_name] = _to_python_scalar(
                                    getter()
                                )
                            except TypeError:
                                continue
                            except Exception as exc:
                                runtime_attrs[link_name][attr_name] = (
                                    f"<error: {exc}>"
                                )

        try:
            stiffness, damping, max_effort, max_velocity, friction = asset.get_joint_drive()
            runtime_attrs["__joint_drive__"] = {
                "stiffness": stiffness[0].detach().cpu().tolist(),
                "damping": damping[0].detach().cpu().tolist(),
                "max_effort": max_effort[0].detach().cpu().tolist(),
                "max_velocity": max_velocity[0].detach().cpu().tolist(),
                "friction": friction[0].detach().cpu().tolist(),
            }
        except Exception as exc:
            runtime_attrs["__joint_drive__"] = {"error": str(exc)}
    except Exception as exc:
        logger.log_warning(
            f"[DEBUG][Articulation:{entity_cfg.uid}] runtime physical query failed: {exc}"
        )

    if len(runtime_attrs) == 0:
        logger.log_warning(
            f"[DEBUG][Articulation:{entity_cfg.uid}] no runtime physical attributes were collected."
        )
        return

    # Explicitly print mass for every link as requested.
    total_mass = 0.0
    has_mass = False
    for link_name, attr_dict in runtime_attrs.items():
        if not isinstance(attr_dict, dict):
            continue
        if "get_mass" in attr_dict:
            has_mass = True
            if isinstance(attr_dict["get_mass"], (int, float)):
                total_mass += float(attr_dict["get_mass"])
            logger.log_info(
                f"[DEBUG][Articulation:{entity_cfg.uid}] {link_name}.mass = {attr_dict['get_mass']}",
                color="green",
            )

    if has_mass:
        logger.log_info(
            f"[DEBUG][Articulation:{entity_cfg.uid}] total_mass = {total_mass}",
            color="green",
        )

    logger.log_info(
        f"[DEBUG][Articulation:{entity_cfg.uid}] runtime physical attrs:\n{json.dumps(runtime_attrs, indent=2, default=str)}",
        color="green",
    )



def visualize_collision_bodies(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_uids: list[str] | str,
    visible: bool = True,
    rgba: list[float] | tuple[float, float, float, float] | None = None,
    link_names_map: Dict[str, List[str]] | None = None,
    control_part_map: Dict[str, str] | None = None,
):
    """Toggle collision-body rendering for selected entities.

    Args:
        entity_uids: Entity uid list or alias string (e.g. "all_objects").
        visible: Whether to show collision bodies.
        rgba: Optional RGBA color for collision-body visualization.
        link_names_map: Optional articulation link-name mapping by uid.
        control_part_map: Optional robot control-part mapping by uid.
    """
    resolved_uids = resolve_uids(env, entity_uids)
    link_names_map = {} if link_names_map is None else link_names_map
    control_part_map = {} if control_part_map is None else control_part_map

    for uid in resolved_uids:
        asset = env.sim.get_asset(uid)
        if asset is None:
            logger.log_warning(
                f"Cannot visualize collision body: asset '{uid}' not found."
            )
            continue

        if isinstance(asset, (RigidObject, RigidObjectGroup)):
            asset.set_physical_visible(visible=visible, rgba=rgba)
        elif isinstance(asset, Articulation):
            asset.set_physical_visible(visible, link_names_map.get(uid, None), rgba)
        elif isinstance(asset, Robot):
            asset.set_physical_visible(visible, control_part_map.get(uid, None), rgba)
        else:
            logger.log_warning(
                f"Asset '{uid}' with type {type(asset)} does not support collision-body visualization."
            )


def visualize_affordance_pose(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    pose_key: str,
    marker_name: str = "debug_pose_marker",
    axis_size: float = 0.003,
    axis_len: float = 0.06,
    arena_index: int = 0,
    remove_old: bool = True,
):
    """Visualize a 4x4 pose from env.affordance_datas in the simulation window.

    This is useful for debugging generated affordance poses in preview/video rendering.
    """
    from embodichain.lab.sim.cfg import MarkerCfg

    pose = env.affordance_datas.get(pose_key, None)
    if pose is None:
        logger.log_warning(
            f"Cannot visualize pose: key '{pose_key}' not found in env.affordance_datas."
        )
        return

    if isinstance(pose, torch.Tensor):
        pose_np = pose.detach().cpu().numpy()
    else:
        pose_np = np.asarray(pose)

    # Use the first env pose if batched as (N, 4, 4).
    if pose_np.ndim == 3:
        pose_np = pose_np[0]

    if pose_np.shape != (4, 4):
        logger.log_warning(
            f"Cannot visualize pose key '{pose_key}': expected shape (4, 4), got {pose_np.shape}."
        )
        return

    marker_storage_name = (
        f"{marker_name}_{arena_index}" if arena_index >= 0 else marker_name
    )
    if remove_old:
        marker_map = getattr(env.sim, "_markers", None)
        if isinstance(marker_map, dict) and marker_storage_name in marker_map:
            env.sim.remove_marker(marker_storage_name)

    env.sim.draw_marker(
        cfg=MarkerCfg(
            name=marker_name,
            marker_type="axis",
            axis_xpos=pose_np,
            axis_size=axis_size,
            axis_len=axis_len,
            arena_index=arena_index,
        )
    )


def visualize_rigid_body_pose(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    marker_name: str = "debug_rigid_pose_marker",
    axis_size: float = 0.003,
    axis_len: float = 0.06,
    arena_index: int = 0,
    remove_old: bool = True,
):
    """Visualize a rigid body's coordinate frame as an axis marker.

    This can be used in reset/interval events to keep rendering a rigid body pose.
    """
    from embodichain.lab.sim.cfg import MarkerCfg

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if isinstance(entity_cfg, dict):
        entity_cfg = SceneEntityCfg(**entity_cfg)

    asset = env.sim.get_asset(entity_cfg.uid)
    if not isinstance(asset, RigidObject):
        logger.log_warning(
            f"visualize_rigid_body_pose only supports RigidObject. Got uid='{entity_cfg.uid}', type={type(asset)}."
        )
        return

    if len(env_ids) == 0:
        logger.log_warning("No env_ids provided for visualize_rigid_body_pose.")
        return

    pose = asset.get_local_pose(to_matrix=True)[env_ids, :]
    pose_np = pose.detach().cpu().numpy() if isinstance(pose, torch.Tensor) else np.asarray(pose)

    marker_storage_name = (
        f"{marker_name}_{arena_index}" if arena_index >= 0 else marker_name
    )
    if remove_old:
        marker_map = getattr(env.sim, "_markers", None)
        if isinstance(marker_map, dict) and marker_storage_name in marker_map:
            env.sim.remove_marker(marker_storage_name)

    if arena_index >= 0:
        env_ids_list = env_ids.detach().cpu().tolist()
        if arena_index in env_ids_list:
            pose_np = pose_np[env_ids_list.index(arena_index)]
        else:
            pose_np = pose_np[0]
            logger.log_warning(
                f"Arena index {arena_index} not found in env_ids {env_ids_list}. Using first env pose."
            )

    env.sim.draw_marker(
        cfg=MarkerCfg(
            name=marker_name,
            marker_type="axis",
            axis_xpos=pose_np,
            axis_size=axis_size,
            axis_len=axis_len,
            arena_index=arena_index,
        )
    )