
import os
import dexsim
import open3d as o3d

from typing import List, Union

from dexsim.types import (
    DriveType,
    ArticulationFlag,
    LoadOption,
    RigidBodyShape,
    SDFConfig,
    PhysicalAttr,
)
from dexsim.engine import Articulation
from dexsim.environment import Env, Arena
from dexsim.models import MeshObject

from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg, SoftObjectCfg
from embodichain.lab.sim.shapes import MeshCfg, CubeCfg, SphereCfg
from embodichain.utils import logger
from dexsim.kit.meshproc import get_mesh_auto_uv
import numpy as np


def load_mesh_objects_from_cfg(
    cfg: RigidObjectCfg, env_list: List[Arena], cache_dir: str | None = None
) -> List[MeshObject]:
    """Load mesh objects from configuration.

    Args:
        cfg (RigidObjectCfg): Configuration for the rigid object.
        env_list (List[Arena]): List of arenas to load the objects into.

    cache_dir (str | None, optional): Directory for caching convex decomposition files. Defaults to None
    Returns:
        List[MeshObject]: List of loaded mesh objects.
    """
    obj_list = []
    body_type = cfg.to_dexsim_body_type()
    if isinstance(cfg.shape, MeshCfg):

        option = LoadOption()
        option.rebuild_normals = cfg.shape.load_option.rebuild_normals
        option.rebuild_tangent = cfg.shape.load_option.rebuild_tangent
        option.rebuild_3rdnormal = cfg.shape.load_option.rebuild_3rdnormal
        option.rebuild_3rdtangent = cfg.shape.load_option.rebuild_3rdtangent
        option.smooth = cfg.shape.load_option.smooth

        cfg: RigidObjectCfg
        fpath = cfg.shape.fpath

        compute_uv = cfg.shape.compute_uv

        is_usd = fpath.endswith((".usd", ".usda", ".usdc"))
        if is_usd:
            # TODO: Currently add checking for num_envs when file is USD. After we support spawn via cloning, we can remove this.
            if len(env_list) > 1:
                logger.log_error(f"Currently not supporting multiple arenas for USD.")
            _env: dexsim.environment.Env = dexsim.default_world().get_env()
            results = _env.import_from_usd_file(fpath, return_object=True)
            # print(f"import usd result: {results}")

            rigidbodys_found = []
            for key, value in results.items():
                if isinstance(value, MeshObject):
                    rigidbodys_found.append(value)
            if len(rigidbodys_found) == 0:
                logger.log_error(f"No rigid body found in USD file: {fpath}")
            elif len(rigidbodys_found) > 1:
                logger.log_error(f"Multiple rigid bodies found in USD file: {fpath}.")
            elif len(rigidbodys_found) == 1:
                obj_list.append(rigidbodys_found[0])
                return obj_list
        else:
            # non-usd file does not support this option, will be forced set False to avoid potential issues.
            cfg.use_usd_properties = False

        for i, env in enumerate(env_list):
            if cfg.sdf_resolution > 0:
                obj = env.load_actor(
                    fpath, duplicate=True, attach_scene=True, option=option
                )
                sdf_cfg = SDFConfig()
                sdf_cfg.resolution = cfg.sdf_resolution
                obj.add_physical_body(
                    body_type,
                    RigidBodyShape.SDF,
                    config=sdf_cfg,
                    attr=PhysicalAttr(),
                )
            else:
                obj = env.load_actor(
                    fpath, duplicate=True, attach_scene=True, option=option
                )
                obj.add_rigidbody(body_type, RigidBodyShape.CONVEX)
            obj.set_name(f"{cfg.uid}_{i}")
            obj_list.append(obj)

            if compute_uv:
                vertices = obj.get_vertices()
                triangles = obj.get_triangles()

                o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
                _, uvs = get_mesh_auto_uv(
                    o3d_mesh, np.array(cfg.shape.project_direction)
                )
                obj.set_uv_mapping(uvs)

    elif isinstance(cfg.shape, CubeCfg):
        from embodichain.lab.sim.utility.sim_utils import create_cube

        obj_list = create_cube(env_list, cfg.shape.size, uid=cfg.uid)
        for obj in obj_list:
            obj.add_rigidbody(body_type, RigidBodyShape.BOX)

    elif isinstance(cfg.shape, SphereCfg):
        from embodichain.lab.sim.utility.sim_utils import create_sphere

        obj_list = create_sphere(
            env_list, cfg.shape.radius, cfg.shape.resolution, uid=cfg.uid
        )
        for obj in obj_list:
            obj.add_rigidbody(body_type, RigidBodyShape.SPHERE)
    else:
        logger.log_error(
            f"Unsupported rigid object shape type: {type(cfg.shape)}. Supported types: MeshCfg, CubeCfg, SphereCfg."
        )
    return obj_list

