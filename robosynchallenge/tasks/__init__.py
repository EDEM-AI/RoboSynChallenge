# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

"""Public task classes backed by the native Task Program deployments."""

from .click_bell import (
    ClickBellEnv,
    ClickBellTestEnv,
    ClickBellAgentEnv,
)
from .drawer_open_place import (
    DrawerOpenPlaceEnv,
    DrawerOpenPlaceTestEnv,
    DrawerOpenPlaceAgentEnv,
)
from .handle_basket import (
    HandleBasketEnv,
    HandleBasketTestEnv,
    HandleBasketAgentEnv,
)
from .item_assembly import (
    ItemAssemblyEnv,
    ItemAssemblyAgentEnv,
)
from .items_handover import (
    ItemsHandoverEnv,
    ItemsHandoverTestEnv,
    ItemsHandoverAgentEnv,
)
from .manipulate_pipette import (
    ManipulatePipetteEnv,
    ManipulatePipetteTestEnv,
    ManipulatePipetteAgentEnv,
)
from .mixer_operating import (
    MixerOperatingEnv,
    MixerOperatingTestEnv,
    MixerOperatingAgentEnv,
)
from .sample_loading import (
    SampleLoadingEnv,
    SampleLoadingTestEnv,
    SampleLoadingAgentEnv,
)
from .table_rearrangement import (
    TableRearrangementEnv,
    TableRearrangementTestEnv,
)
from .water_pouring import (
    WaterPouringEnv,
    WaterPouringTestEnv,
    WaterPouringAgentEnv,
)
from ._other_tasks.manipulate_pipette_two_beaker import (
    ManipulatePipetteTwoBeakerEnv,
    ManipulatePipetteTwoBeakerTestEnv,
    ManipulatePipetteTwoBeakerAgentEnv,
)
from ._other_tasks.open_pan import (
    OpenPanPickAndPlaceEnv,
    OpenPanPickAndPlaceTestEnv,
    OpenPanPickAndPlaceAgentEnv,
)
from ._other_tasks.pour_water import (
    PourWaterEnv,
    PourWaterAgentEnv,
)
from ._other_tasks.sample_loading import SampleLoadingEnv as SingleArmSampleLoadingEnv
