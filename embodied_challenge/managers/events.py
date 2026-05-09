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
from embodichain.lab.gym.envs.managers.events import resolve_uids
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
    matrix_from_euler,
    matrix_from_quat,
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


def _cfg_get(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _iter_event_cfgs(events_cfg):
    if events_cfg is None:
        return []
    if isinstance(events_cfg, dict):
        return list(events_cfg.items())
    if hasattr(events_cfg, "items"):
        return list(events_cfg.items())
    if hasattr(events_cfg, "__dict__"):
        return list(vars(events_cfg).items())
    return []


def _callable_name(func) -> str:
    if isinstance(func, str):
        return func.rsplit(".", 1)[-1]
    return getattr(func, "__name__", func.__class__.__name__)


def _resolve_entity_uid(entity_cfg=None, uid: str | None = None) -> str | None:
    if uid is not None:
        return uid
    if entity_cfg is None:
        return None
    if isinstance(entity_cfg, str):
        return entity_cfg
    if isinstance(entity_cfg, dict):
        return entity_cfg.get("uid", None)
    return getattr(entity_cfg, "uid", None)


def _resolve_entity_uids_from_params(params) -> list[str]:
    uids = []
    single_uid = _resolve_entity_uid(_cfg_get(params, "entity_cfg", None))
    if single_uid is not None:
        uids.append(single_uid)

    entity_cfgs = _cfg_get(params, "entity_cfgs", None)
    if entity_cfgs is None:
        return uids
    if isinstance(entity_cfgs, (str, dict, SceneEntityCfg)):
        entity_cfgs = [entity_cfgs]

    for entity_cfg in entity_cfgs:
        uid = _resolve_entity_uid(entity_cfg)
        if uid is not None:
            uids.append(uid)
    return uids


def _normalize_event_env_ids(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | slice | list[int] | tuple[int, ...] | None,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


def _as_scene_entity_cfgs(
    entity_cfgs: List[SceneEntityCfg] | List[Dict] | None = None,
    entity_cfg: SceneEntityCfg | Dict | None = None,
    entity_uids: List[str] | str | None = None,
) -> list[SceneEntityCfg]:
    if entity_cfgs is None:
        if entity_cfg is not None:
            entity_cfgs = [entity_cfg]
        elif entity_uids is not None:
            if isinstance(entity_uids, str):
                entity_uids = [entity_uids]
            entity_cfgs = [SceneEntityCfg(uid=uid) for uid in entity_uids]
        else:
            logger.log_error(
                "randomize_entity_root_pose_group requires entity_cfgs, entity_cfg, or entity_uids."
            )

    resolved_cfgs = []
    for cfg in entity_cfgs:
        if isinstance(cfg, SceneEntityCfg):
            resolved_cfgs.append(cfg)
        elif isinstance(cfg, dict):
            resolved_cfgs.append(SceneEntityCfg(**cfg))
        elif isinstance(cfg, str):
            resolved_cfgs.append(SceneEntityCfg(uid=cfg))
        else:
            logger.log_error(f"Unsupported entity cfg type: {type(cfg)}.")
    return resolved_cfgs


def _get_asset_initial_root_pose(
    env: EmbodiedEnv,
    asset: RigidObject | Articulation,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_instance = len(env_ids)
    asset_cfg = getattr(asset, "cfg", None)
    init_pos = getattr(asset_cfg, "init_pos", None)
    init_rot = getattr(asset_cfg, "init_rot", None)

    if init_pos is not None and init_rot is not None:
        init_pos = torch.as_tensor(init_pos, dtype=torch.float32, device=env.device)
        if init_pos.ndim == 1:
            init_pos = init_pos.unsqueeze(0).repeat(num_instance, 1)
        else:
            init_pos = init_pos[env_ids]

        init_rot = torch.as_tensor(init_rot, dtype=torch.float32, device=env.device)
        if init_rot.ndim == 1:
            init_rot = init_rot.unsqueeze(0).repeat(num_instance, 1)
        else:
            init_rot = init_rot[env_ids]
        init_rot = matrix_from_euler(init_rot * torch.pi / 180.0)
        return init_pos, init_rot

    current_pose = asset.get_local_pose()[env_ids]
    if current_pose.ndim == 2 and current_pose.shape[-1] == 7:
        return current_pose[:, :3], matrix_from_quat(current_pose[:, 3:7])
    if current_pose.ndim == 3 and current_pose.shape[-2:] == (4, 4):
        return current_pose[:, :3, 3], current_pose[:, :3, :3]

    logger.log_error(
        f"Cannot infer root pose for asset '{asset.cfg.uid}' from cfg or current pose."
    )


def randomize_entity_root_pose_group(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfgs: List[SceneEntityCfg] | List[Dict] | None = None,
    entity_cfg: SceneEntityCfg | Dict | None = None,
    entity_uids: List[str] | str | None = None,
    position_range: tuple[list[float], list[float]] | None = None,
    rotation_range: tuple[list[float], list[float]] | None = None,
    relative_position: bool = True,
    relative_rotation: bool = True,
    physics_update_step: int = -1,
) -> None:
    """Randomize multiple entity root poses with one shared sampled offset.

    The same sampled translation and Euler-angle rotation are reused for every
    entity in ``entity_cfgs``. This keeps paired objects, such as a visual
    background and its physical drawer, aligned during domain randomization.
    ``RigidObject`` and ``Articulation`` assets are supported.
    """
    env_ids = _normalize_event_env_ids(env, env_ids)
    entity_cfgs = _as_scene_entity_cfgs(
        entity_cfgs=entity_cfgs,
        entity_cfg=entity_cfg,
        entity_uids=entity_uids,
    )

    num_instance = len(env_ids)
    if num_instance == 0 or len(entity_cfgs) == 0:
        return

    pos_sample = None
    if position_range is not None:
        pos_sample = sample_uniform(
            lower=torch.tensor(position_range[0], dtype=torch.float32, device=env.device),
            upper=torch.tensor(position_range[1], dtype=torch.float32, device=env.device),
            size=(num_instance, 3),
            device=env.device,
        )

    rot_sample = None
    if rotation_range is not None:
        rot_euler = (
            sample_uniform(
                lower=torch.tensor(rotation_range[0], dtype=torch.float32, device=env.device),
                upper=torch.tensor(rotation_range[1], dtype=torch.float32, device=env.device),
                size=(num_instance, 3),
                device=env.device,
            )
            * torch.pi
            / 180.0
        )
        rot_sample = matrix_from_euler(rot_euler)

    changed = False
    for cfg in entity_cfgs:
        asset = env.sim.get_asset(cfg.uid)
        if not isinstance(asset, (RigidObject, Articulation)):
            logger.log_warning(
                f"randomize_entity_root_pose_group only supports RigidObject or "
                f"Articulation. Got uid='{cfg.uid}', type={type(asset)}. Skipping."
            )
            continue

        init_pos, init_rot = _get_asset_initial_root_pose(env, asset, env_ids)

        pose = (
            torch.eye(4, dtype=torch.float32, device=env.device)
            .unsqueeze(0)
            .repeat(num_instance, 1, 1)
        )
        pose[:, :3, 3] = init_pos
        pose[:, :3, :3] = init_rot

        if pos_sample is not None:
            pose[:, :3, 3] = init_pos + pos_sample if relative_position else pos_sample
        if rot_sample is not None:
            pose[:, :3, :3] = (
                torch.bmm(init_rot, rot_sample) if relative_rotation else rot_sample
            )

        asset.set_local_pose(pose, env_ids=env_ids)
        asset.clear_dynamics(env_ids=env_ids)
        changed = True

    if changed and physics_update_step > 0:
        env.sim.update(step=physics_update_step)


def _find_entity_position_randomization_event(
    env: EmbodiedEnv,
    uid: str,
):
    randomization_func_names = {
        "randomize_rigid_object_pose",
        "randomize_articulation_root_pose",
        "randomize_entity_root_pose_group",
    }
    events_cfg = getattr(env.cfg, "events", None)

    matches = []
    for event_name, event_cfg in _iter_event_cfgs(events_cfg):
        func_name = _callable_name(_cfg_get(event_cfg, "func", None))
        if func_name not in randomization_func_names:
            continue
        params = _cfg_get(event_cfg, "params", {}) or {}
        event_uids = _resolve_entity_uids_from_params(params)
        if uid is not None and uid not in event_uids:
            continue
        if _cfg_get(params, "position_range", None) is None:
            continue
        matches.append((event_name, event_cfg, params))

    if len(matches) > 1:
        match_names = [name for name, _, _ in matches]
        logger.log_warning(
            f"Found multiple position randomization events for uid='{uid}'. "
            f"Using first matched event '{matches[0][0]}'. Matched events: {match_names}."
        )
    return matches[0] if matches else None


def _get_entity_init_position(
    env: EmbodiedEnv,
    uid: str,
    arena_index: int = 0,
) -> torch.Tensor:
    asset = env.sim.get_asset(uid)
    if asset is None:
        logger.log_error(f"Cannot visualize position range: asset '{uid}' not found.")

    asset_cfg = getattr(asset, "cfg", None)
    init_pos = getattr(asset_cfg, "init_pos", None)
    if init_pos is not None:
        return torch.as_tensor(init_pos, dtype=torch.float32, device=env.device)

    try:
        pose = asset.get_local_pose(to_matrix=True)
        if isinstance(pose, torch.Tensor):
            pose = pose[min(max(arena_index, 0), pose.shape[0] - 1)]
            return pose[:3, 3].to(device=env.device, dtype=torch.float32)
    except TypeError:
        pass

    pose = asset.get_local_pose()
    if isinstance(pose, torch.Tensor):
        pose = pose[min(max(arena_index, 0), pose.shape[0] - 1)]
        return pose[:3].to(device=env.device, dtype=torch.float32)

    pose = np.asarray(pose)
    if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
        index = min(max(arena_index, 0), pose.shape[0] - 1)
        return torch.as_tensor(pose[index, :3, 3], dtype=torch.float32, device=env.device)
    if pose.ndim == 2 and pose.shape[-1] >= 3:
        index = min(max(arena_index, 0), pose.shape[0] - 1)
        return torch.as_tensor(pose[index, :3], dtype=torch.float32, device=env.device)

    logger.log_error(
        f"Cannot infer init position for uid='{uid}' from asset cfg or current pose."
    )


def _make_position_range_corner_points(
    low: torch.Tensor,
    high: torch.Tensor,
    table_height: float,
    include_center: bool = False,
) -> torch.Tensor:
    bbox_low = torch.minimum(low, high)
    bbox_high = torch.maximum(low, high)
    z_value = torch.as_tensor(table_height, dtype=bbox_low.dtype, device=bbox_low.device)
    values = [
        [bbox_low[0], bbox_high[0]],
        [bbox_low[1], bbox_high[1]],
        [z_value],
    ]

    corners = []
    seen = set()
    for x in values[0]:
        for y in values[1]:
            for z in values[2]:
                point = torch.stack([x, y, z])
                key = tuple(round(float(v), 8) for v in point.detach().cpu().tolist())
                if key in seen:
                    continue
                seen.add(key)
                corners.append(point)

    if include_center:
        center = torch.stack(
            [
                (bbox_low[0] + bbox_high[0]) * 0.5,
                (bbox_low[1] + bbox_high[1]) * 0.5,
                z_value,
            ]
        )
        key = tuple(round(float(v), 8) for v in center.detach().cpu().tolist())
        if key not in seen:
            corners.append(center)

    return torch.stack(corners, dim=0)


def visualize_entity_position_range(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    uid: str,
    table_height: float,
    marker_name: str | None = None,
    axis_size: float = 0.003,
    axis_len: float = 0.05,
    corner_mode: str = "xy",
    include_center: bool = False,
    arena_index: int = 0,
    remove_old: bool = True,
    verbose: bool = False,
):
    """Visualize the configured position-randomization range for a rigid/articulation uid.

    The function only needs a uid. It searches env.cfg.events for the first matching
    randomize_rigid_object_pose/randomize_articulation_root_pose event, reads its
    position_range and relative_position flag, and draws axes at the XY corners of
    the actual sampled range. The marker z position is always table_height. When
    relative_position is true, the configured offset range is added to the asset's
    initial position before extracting the XY bounds.
    """
    from embodichain.lab.sim.cfg import MarkerCfg

    _ = env_ids
    _ = corner_mode

    matched_event = _find_entity_position_randomization_event(env=env, uid=uid)
    if matched_event is None:
        logger.log_error(
            f"Cannot visualize position range for uid='{uid}': no matching "
            "randomize_rigid_object_pose/randomize_articulation_root_pose event with "
            "position_range was found in env.cfg.events."
        )

    event_name, _, params = matched_event
    position_range = _cfg_get(params, "position_range", None)
    relative_position = bool(_cfg_get(params, "relative_position", True))

    asset = env.sim.get_asset(uid)
    if not isinstance(asset, (RigidObject, Articulation)):
        logger.log_warning(
            f"Position range visualization supports RigidObject or Articulation. "
            f"Got uid='{uid}', type={type(asset)}."
        )
        return None

    range_tensor = torch.as_tensor(
        position_range, dtype=torch.float32, device=env.device
    )
    if range_tensor.shape != (2, 3):
        logger.log_error(
            f"position_range for uid='{uid}' must have shape (2, 3), got {tuple(range_tensor.shape)}."
        )

    low = range_tensor[0]
    high = range_tensor[1]
    if relative_position:
        base_pos = _get_entity_init_position(
            env=env,
            uid=uid,
            arena_index=arena_index,
        )
        low = low + base_pos
        high = high + base_pos

    points = _make_position_range_corner_points(
        low=low,
        high=high,
        table_height=table_height,
        include_center=include_center,
    )

    poses = torch.eye(4, dtype=torch.float32, device=env.device).unsqueeze(0)
    poses = poses.repeat(points.shape[0], 1, 1)
    poses[:, :3, 3] = points

    if marker_name is None:
        marker_name = f"{uid}_position_range_axis"

    marker_storage_name = (
        f"{marker_name}_{arena_index}" if arena_index >= 0 else marker_name
    )
    if remove_old:
        marker_map = getattr(env.sim, "_markers", None)
        if isinstance(marker_map, dict) and marker_storage_name in marker_map:
            env.sim.remove_marker(marker_storage_name)

    if verbose:
        event_part = f" from event '{event_name}'" if event_name is not None else ""
        logger.log_info(
            f"Visualizing position range for uid='{uid}'{event_part}: "
            f"low={low.detach().cpu().tolist()}, high={high.detach().cpu().tolist()}, "
            f"table_height={table_height}, corner_count={points.shape[0]}",
            color="green",
        )

    return env.sim.draw_marker(
        cfg=MarkerCfg(
            name=marker_name,
            marker_type="axis",
            axis_xpos=poses.detach().cpu().numpy(),
            axis_size=axis_size,
            axis_len=axis_len,
            arena_index=arena_index,
        )
    )

#####Distractor库实现##############################
class replace_distractor_slots_from_library(Functor):
    """Replace distractor slots with random assets from a library on each reset.

    This functor is designed for rapid dataset curation where distractor assets can be
    swapped every episode. It pre-loads all assets during initialization, keeps them
    hidden below the table, and only moves two selected assets into the visible slots
    on each call to avoid dynamic remove/add that causes GPU sync issues.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)

        self._asset_paths = self._load_asset_paths(cfg.params)
        if len(self._asset_paths) == 0:
            logger.log_error(
                "No distractor assets found. Please check library_index_path/folder_path and filters."
            )
        self._hide_position = tuple(cfg.params.get("hide_position", [0.0, 0.0, -20.0]))
        self._hide_spacing = float(cfg.params.get("hide_spacing", 0.5))
        self._disable_hidden_collisions = bool(
            cfg.params.get("disable_hidden_collisions", True)
        )
        self._toggle_visibility_on_reset = bool(
            cfg.params.get("toggle_visibility_on_reset", False)
        )
        self._pool_uids: list[str] = []
        self._slot_uids = self._resolve_slot_uids(cfg.params.get("entity_cfgs", []))

        self._preload_asset_pool(env, cfg.params)
        self._hide_scene_objects(
            env,
            self._pool_uids,
            update_visibility=self._toggle_visibility_on_reset,
        )
        self._hide_scene_objects(env, self._slot_uids, update_visibility=True)

    def _resolve_slot_uids(self, entity_cfgs: List[SceneEntityCfg] | List[Dict]) -> list[str]:
        uids = []
        for entity_cfg in entity_cfgs:
            if isinstance(entity_cfg, dict):
                uid = entity_cfg.get("uid", None)
            else:
                uid = getattr(entity_cfg, "uid", None)
            if uid is not None:
                uids.append(uid)
        return uids

    def _resolve_local_or_data_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path

        # Try data registry resolution first, then workspace-relative fallback.
        try:
            return get_data_path(path)
        except Exception:
            return os.path.abspath(path)

    def _get_hidden_pose(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor,
        hide_index: int = 0,
    ) -> torch.Tensor:
        hidden_pose = torch.zeros((len(env_ids), 7), device=env.device)
        hidden_pose[:, :3] = torch.tensor(
            self._compute_hidden_position(hide_index), device=env.device
        )
        hidden_pose[:, 3] = 1.0
        return hidden_pose

    def _compute_hidden_position(self, hide_index: int = 0) -> tuple[float, float, float]:
        hidden_pos = list(self._hide_position)
        if hide_index > 0 and self._hide_spacing > 0.0:
            grid_index = hide_index - 1
            hidden_pos[0] += (grid_index % 8 + 1) * self._hide_spacing
            hidden_pos[1] += (grid_index // 8) * self._hide_spacing
        return tuple(hidden_pos)

    def _set_collision_enabled(
        self,
        obj: RigidObject,
        env: EmbodiedEnv,
        env_ids: torch.Tensor,
        enabled: bool | torch.Tensor,
    ) -> None:
        if not self._disable_hidden_collisions or not hasattr(obj, "enable_collision"):
            return

        if isinstance(enabled, torch.Tensor):
            enable_mask = enabled.to(device=env.device, dtype=torch.bool)
        else:
            enable_mask = torch.full(
                (len(env_ids),), bool(enabled), dtype=torch.bool, device=env.device
            )

        collision_env_ids = (
            env_ids.detach().cpu().tolist()
            if isinstance(env_ids, torch.Tensor)
            else env_ids
        )
        obj.enable_collision(enable_mask, env_ids=collision_env_ids)

    def _hide_scene_objects(
        self,
        env: EmbodiedEnv,
        uids: list[str],
        env_ids: torch.Tensor | None = None,
        update_visibility: bool = False,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        for hide_index, uid in enumerate(uids):
            hidden_pose = self._get_hidden_pose(env, env_ids, hide_index=hide_index)
            obj = env.sim.get_rigid_object(uid)
            if obj is None:
                continue
            if update_visibility:
                obj.set_visible(False)
            obj.set_local_pose(hidden_pose, env_ids=env_ids)
            self._set_collision_enabled(obj, env, env_ids, False)
            obj.clear_dynamics(env_ids=env_ids)

    def _preload_asset_pool(self, env: EmbodiedEnv, params: Dict) -> None:
        entity_cfgs = params.get("entity_cfgs", [])
        if len(entity_cfgs) == 0:
            logger.log_error("entity_cfgs must be provided for distractor slots.")

        template_uid = self._slot_uids[0] if len(self._slot_uids) > 0 else None
        if template_uid is None:
            logger.log_error("At least one distractor slot is required as template.")

        template_asset = env.sim.get_asset(template_uid)
        if template_asset is None or not isinstance(template_asset, RigidObject):
            logger.log_error(
                f"Template distractor slot '{template_uid}' must be a RigidObject."
            )

        template_cfg = deepcopy(template_asset.cfg)

        for idx, asset_path in enumerate(self._asset_paths):
            base_name = os.path.splitext(os.path.basename(asset_path))[0]
            pool_uid = f"distractor_pool_{idx:03d}_{base_name}"
            if env.sim.get_asset(pool_uid) is not None:
                logger.log_error(f"Duplicated distractor pool uid generated: {pool_uid}")

            pool_cfg = deepcopy(template_cfg)
            pool_cfg.uid = pool_uid
            pool_cfg.shape.fpath = asset_path
            pool_cfg.init_pos = self._compute_hidden_position(idx)
            pool_cfg.init_rot = (0.0, 0.0, 0.0)

            pool_obj = env.sim.add_rigid_object(cfg=pool_cfg)
            # Keep pool objects renderable and hide them by pose. Repeated
            # visibility toggles can touch native renderer state every reset.
            pool_obj.set_visible(not self._toggle_visibility_on_reset)
            self._pool_uids.append(pool_uid)

    def _load_asset_paths(self, params: Dict) -> list[str]:
        exts = params.get("exts", [".ply", ".obj", ".stl", ".glb", ".gltf"])

        index_path = params.get("library_index_path", None)
        if index_path is not None:
            index_path = self._resolve_local_or_data_path(index_path)
            if not os.path.exists(index_path):
                logger.log_error(f"library_index_path not found: {index_path}")
            with open(index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            categories = params.get("categories", None)
            all_categories: Dict[str, list[str]] = payload.get("categories", {})

            if categories is None:
                selected = all_categories
            else:
                selected = {
                    key: value
                    for key, value in all_categories.items()
                    if key in set(categories)
                }

            index_dir = os.path.dirname(index_path)
            asset_paths = []
            for rel_list in selected.values():
                for rel_path in rel_list:
                    abs_path = os.path.join(index_dir, rel_path)
                    if os.path.isfile(abs_path):
                        asset_paths.append(abs_path)

            if len(exts) > 0:
                lower_exts = tuple(ext.lower() for ext in exts)
                asset_paths = [
                    path for path in asset_paths if path.lower().endswith(lower_exts)
                ]

            return sorted(list(set(asset_paths)))

        folder_path = params.get("folder_path", None)
        if folder_path is None:
            logger.log_error("Either library_index_path or folder_path must be provided.")

        folder_path = self._resolve_local_or_data_path(folder_path)
        if not os.path.isdir(folder_path):
            logger.log_error(f"folder_path not found or not a directory: {folder_path}")

        patterns = params.get("patterns", None)
        asset_paths = get_all_files_in_directory(
            folder_path, exts=exts, patterns=patterns
        )
        return sorted(list(set(asset_paths)))

    def _sample_positions(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor,
        position_range: Tuple[List[float], List[float]],
        avoid_uids: list[str],
        min_distance_to_avoid: float,
        max_attempts: int,
    ) -> torch.Tensor:
        num_envs = len(env_ids)
        low = torch.tensor(position_range[0], device=env.device)
        high = torch.tensor(position_range[1], device=env.device)

        pos = sample_uniform(
            lower=low,
            upper=high,
            size=(num_envs, 3),
            device=env.device,
        )

        if len(avoid_uids) == 0 or min_distance_to_avoid <= 0.0:
            return pos

        ref_positions = []
        for uid in avoid_uids:
            asset = env.sim.get_asset(uid)
            if asset is None:
                continue
            ref_positions.append(asset.get_local_pose(to_matrix=True)[env_ids, :2, 3])

        if len(ref_positions) == 0:
            return pos

        threshold2 = float(min_distance_to_avoid) * float(min_distance_to_avoid)
        valid_mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

        for _ in range(max_attempts):
            dist_ok = torch.ones(num_envs, dtype=torch.bool, device=env.device)
            for ref_pos in ref_positions:
                delta = pos[..., :2] - ref_pos[..., :2]
                d2 = torch.sum(delta * delta, dim=-1)
                dist_ok = torch.logical_and(dist_ok, d2 >= threshold2)

            valid_mask = torch.logical_or(valid_mask, dist_ok)
            if bool(torch.all(valid_mask)):
                break

            need = (~valid_mask).nonzero(as_tuple=False).squeeze(-1)
            resampled = sample_uniform(
                lower=low,
                upper=high,
                size=(len(need), 3),
                device=env.device,
            )
            pos[need] = resampled

        return pos

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor | None,
        entity_cfgs: List[SceneEntityCfg],
        library_index_path: str | None = None,
        categories: List[str] | None = None,
        folder_path: str | None = None,
        patterns: List[str] | None = None,
        exts: List[str] | None = None,
        position_ranges: List[Tuple[List[float], List[float]]] | None = None,
        appear_probs: List[float] | None = None,
        z_rotation_ranges: List[Tuple[float, float]] | None = None,
        hide_position: List[float] = [0.0, 0.0, -20.0],
        hide_spacing: float = 0.5,
        disable_hidden_collisions: bool = True,
        toggle_visibility_on_reset: bool = False,
        avoid_uids: List[str] | None = None,
        min_distance_to_avoid: float = 0.0,
        max_resample_attempts: int = 20,
        physics_update_step: int = 1,
    ) -> None:
        """Select and position distractor assets from the pre-loaded pool.

        Two assets are sampled without replacement on each reset and moved to the
        configured distractor slot ranges. All other pool objects remain hidden below
        the floor.
        """
        # NOTE: library-related kwargs are resolved during functor init from cfg.params.
        # They are kept in call signature for manager argument validation compatibility.
        _ = (
            library_index_path,
            categories,
            folder_path,
            patterns,
            exts,
            hide_position,
            hide_spacing,
            disable_hidden_collisions,
            toggle_visibility_on_reset,
        )

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        if position_ranges is None:
            logger.log_error("position_ranges must be provided for distractor slots.")

        if appear_probs is None:
            appear_probs = [1.0] * len(entity_cfgs)

        if z_rotation_ranges is None:
            z_rotation_ranges = [(-180.0, 180.0)] * len(entity_cfgs)

        if avoid_uids is None:
            avoid_uids = []

        if (
            len(entity_cfgs) != len(position_ranges)
            or len(entity_cfgs) != len(appear_probs)
            or len(entity_cfgs) != len(z_rotation_ranges)
        ):
            logger.log_error(
                "entity_cfgs/position_ranges/appear_probs/z_rotation_ranges must have same length."
            )

        if len(self._pool_uids) < len(entity_cfgs):
            logger.log_error(
                f"Not enough preloaded distractor assets ({len(self._pool_uids)}) for slots ({len(entity_cfgs)})."
            )

        selected_uids = random.sample(self._pool_uids, k=len(entity_cfgs))

        self._hide_scene_objects(
            env,
            self._pool_uids,
            env_ids=env_ids,
            update_visibility=self._toggle_visibility_on_reset,
        )
        self._hide_scene_objects(
            env,
            self._slot_uids,
            env_ids=env_ids,
            update_visibility=self._toggle_visibility_on_reset,
        )

        for idx, uid in enumerate(selected_uids):
            obj = env.sim.get_rigid_object(uid)
            if obj is None:
                logger.log_error(f"Preloaded distractor asset '{uid}' not found in scene.")
                continue

            appear_prob = float(appear_probs[idx])
            appear_mask = (
                torch.rand(len(env_ids), device=env.device)
                < max(0.0, min(appear_prob, 1.0))
            )

            pos = self._sample_positions(
                env=env,
                env_ids=env_ids,
                position_range=position_ranges[idx],
                avoid_uids=avoid_uids,
                min_distance_to_avoid=min_distance_to_avoid,
                max_attempts=max_resample_attempts,
            )

            yaw_low, yaw_high = z_rotation_ranges[idx]
            yaw = sample_uniform(
                lower=torch.tensor([yaw_low], device=env.device),
                upper=torch.tensor([yaw_high], device=env.device),
                size=(len(env_ids), 1),
                device=env.device,
            )
            yaw = yaw * torch.pi / 180.0

            pose = torch.zeros((len(env_ids), 7), device=env.device)
            pose[:, :3] = pos
            pose[:, 3] = torch.cos(yaw[:, 0] * 0.5)
            pose[:, 6] = torch.sin(yaw[:, 0] * 0.5)

            final_pose = self._get_hidden_pose(env, env_ids, hide_index=idx)
            final_pose[appear_mask] = pose[appear_mask]

            if self._toggle_visibility_on_reset:
                obj.set_visible(True)
            obj.set_local_pose(final_pose, env_ids=env_ids)
            self._set_collision_enabled(obj, env, env_ids, appear_mask)
            obj.clear_dynamics(env_ids=env_ids)

        if physics_update_step > 0:
            env.sim.update(step=physics_update_step)


#######Distractor库实现###############################
