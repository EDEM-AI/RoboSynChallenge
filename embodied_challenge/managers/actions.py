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

"""Action terms for processing policy actions into robot control commands.

This module provides concrete implementations of :class:`ActionTerm` that convert
raw policy actions into different control formats (e.g., joint positions, velocities,
forces, or end-effector poses).

The action terms are typically used in conjunction with :class:`ActionManager` which
handles calling the appropriate term based on configuration.

Available action terms:

- :class:`DeltaQposTerm`: Delta joint position (current + scale * action)
- :class:`QposTerm`: Absolute joint position (scale * action)
- :class:`QposDenormalizedTerm`: Normalized action [-1,1] -> joint limits
- :class:`EefPoseTerm`: End-effector pose -> IK -> joint position
- :class:`QvelTerm`: Joint velocity (scale * action)
- :class:`QfTerm`: Joint force/torque (scale * action)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from embodichain.lab.sim.types import EnvAction
from embodichain.utils.math import matrix_from_euler, matrix_from_quat
from embodichain.lab.gym.envs.managers.action_manager import ActionTerm
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg

# Import ActionTerm from action_manager after it's defined
# This is a late import to avoid circular dependency
if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


__all__ = [

]

