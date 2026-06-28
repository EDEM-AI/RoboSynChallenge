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

"""Dataset functors for collecting and saving episode data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import gymnasium as gym
import torch
import tqdm

from tensordict import TensorDict

from embodichain.utils import logger
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATASET_ROOT
from embodichain.lab.gym.utils.misc import is_stereocam
from embodichain.lab.sim.sensors import Camera, ContactSensor
from embodichain.lab.gym.envs.managers.manager_base import Functor
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
from embodichain.lab.gym.envs.managers.datasets import (
    LeRobotRecorder as EmbodiChainLeRobotRecorder,
)
from robosynchallenge.data.constants import ROBOSYNCHALLENGE_ROOT

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

__all__ = ["LeRobotRecorder", "install_lerobot_recorder_override"]


class LeRobotRecorder(EmbodiChainLeRobotRecorder):
    """LeRobot recorder with save_path resolved relative to RoboSynChallenge."""

    def __init__(self, cfg: DatasetFunctorCfg, env: EmbodiedEnv):
        save_path = cfg.params.get("save_path", None)
        if save_path:
            save_path = Path(str(save_path)).expanduser()
            if not save_path.is_absolute():
                params = dict(cfg.params)
                params["save_path"] = str(ROBOSYNCHALLENGE_ROOT / save_path)
                cfg.params = params

        super().__init__(cfg, env)


def install_lerobot_recorder_override() -> None:
    """Make EmbodiChain config lookup resolve LeRobotRecorder to this subclass."""
    import embodichain.lab.gym.envs.managers as manager_module
    import embodichain.lab.gym.envs.managers.datasets as dataset_module

    dataset_module.LeRobotRecorder = LeRobotRecorder
    manager_module.LeRobotRecorder = LeRobotRecorder
