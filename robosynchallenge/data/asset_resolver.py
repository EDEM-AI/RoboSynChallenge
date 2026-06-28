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

import os
import sys
from pathlib import Path
from typing import Callable, Optional

import embodichain.data as embodichain_data
import embodichain.data.dataset as embodichain_dataset

from .constants import ROBOSYNCHALLENGE_ROOT


_ORIGINAL_GET_DATA_PATH: Callable[[str], str] = embodichain_dataset.get_data_path


def resolve_local_asset_path(data_path_in_config: str) -> Optional[str]:
    """Resolve a RoboSynChallenge-relative asset path without triggering downloads."""
    if not data_path_in_config:
        return None

    path = Path(os.path.expanduser(data_path_in_config))
    if path.is_absolute():
        return str(path) if path.exists() else None

    candidate = ROBOSYNCHALLENGE_ROOT / path
    if candidate.exists():
        return str(candidate)

    return None


def get_data_path(data_path_in_config: str) -> str:
    """Resolve local RoboSynChallenge assets before falling back to EmbodiChain."""
    if os.path.isabs(data_path_in_config):
        return data_path_in_config

    local_path = resolve_local_asset_path(data_path_in_config)
    if local_path is not None:
        return local_path

    return _ORIGINAL_GET_DATA_PATH(data_path_in_config)


def install_embodichain_asset_resolver() -> None:
    """Patch EmbodiChain path helpers to use RoboSynChallenge local assets first."""
    embodichain_dataset.get_data_path = get_data_path
    embodichain_data.get_data_path = get_data_path

    # Patch modules that may already have imported get_data_path by value.
    for module_name in (
        "embodichain.lab.sim.cfg",
        "embodichain.lab.gym.utils.gym_utils",
        "embodichain.lab.gym.envs.embodied_env",
        "embodichain.lab.gym.envs.managers.events",
        "embodichain.lab.gym.envs.managers.randomization.visual",
        "robosynchallenge.managers.events",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "get_data_path"):
            setattr(module, "get_data_path", get_data_path)
