
import torch
import dexsim
import numpy as np

from dataclasses import dataclass
from typing import List, Sequence, Union

from dexsim.models import MeshObject
from dexsim.types import RigidBodyGPUAPIReadType, RigidBodyGPUAPIWriteType
from dexsim.engine import CudaArray, PhysicsScene
from embodichain.lab.sim.cfg import RigidObjectCfg, RigidBodyAttributesCfg
from embodichain.lab.sim import (
    VisualMaterial,
    VisualMaterialInst,
    BatchEntity,
)
from embodichain.lab.sim.utility import is_rt_enabled
from embodichain.utils.math import convert_quat
from embodichain.utils.math import matrix_from_quat, quat_from_matrix, matrix_from_euler
from embodichain.utils import logger


###class RigidObject(BatchEntity):

def __str__(self) -> str:
    parent_str = super().__str__()
    return (
        parent_str
        + f" | body type: {self.body_type}"
        )
