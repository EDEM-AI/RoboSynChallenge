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
import os

from tensordict import TensorDict
from typing import TYPE_CHECKING, Literal, Union, List, Dict, Sequence

from embodichain.lab.sim.objects import RigidObject, Articulation, Robot
from embodichain.lab.sim.sensors import Camera, StereoCamera
from embodichain.lab.sim.types import EnvObs
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import resolve_dict
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.utils import logger
from embodichain.utils.math import quat_from_matrix, euler_xyz_from_quat

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

