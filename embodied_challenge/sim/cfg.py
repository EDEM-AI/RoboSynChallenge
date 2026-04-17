
from __future__ import annotations
import os
import numpy as np
import torch

from typing import Sequence, Union, Dict, Literal, List, Any, Optional
from dataclasses import field, MISSING

from dexsim.types import (
    PhysicalAttr,
    ActorType,
    AxisArrowType,
    AxisCornerType,
    VoxelConfig,
    SoftBodyAttr,
    SoftBodyMaterialModel,
)
from embodichain.utils import configclass, is_configclass
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data import get_data_path
from embodichain.utils import logger
from embodichain.utils.utility import key_in_nested_dict

from .shapes import ShapeCfg, MeshCfg

@configclass
class RigidObjectCfg(ObjectBaseCfg):
    """Configuration for a rigid body asset in the simulation.

    This class extends the base asset configuration to include specific properties for rigid bodies,
    such as physical attributes and collision group.
    """

    shape: ShapeCfg = ShapeCfg()
    """Shape configuration for the rigid body. """

    # TODO: supoort basic primitive shapes, such as box, sphere, etc cfg and spawn method.

    attrs: RigidBodyAttributesCfg = RigidBodyAttributesCfg()

    body_type: Literal["dynamic", "kinematic", "static"] = "dynamic"
#########max_convex_hull_num#############
    max_convex_hull_num: int = 1
    """Deprecated compatibility field for rigid bodies.

    This value is ignored for rigid object loading. The loader no longer performs ACD
    decomposition for rigid meshes because some assets are unstable with that path.
    """
