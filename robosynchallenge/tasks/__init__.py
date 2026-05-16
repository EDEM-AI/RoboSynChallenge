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

# entry-level tasks
from .click_bell.click_bell import (
    ClickBellEnv,
    ClickBellTestEnv,
    ClickBellAgentEnv,
)
from .handle_basket.handle_basket import (
    HandleBasketEnv,
    HandleBasketAgentEnv,
)
from .water_pouring.water_pouring import (
    WaterPouringEnv,
    WaterPouringTestEnv,
    WaterPouringAgentEnv,
)
from .table_rearrangement.table_rearrangement import (
    TableRearrangementEnv,
    TableRearrangementTestEnv
)


# mid-level tasks
from .items_handover.items_handover import (
    ItemsHandoverEnv,
    ItemsHandoverTestEnv,
)
from .drawer_open_place.drawer_open_place import (
    DrawerOpenPlaceAgentEnv,
    DrawerOpenPlaceEnv,
)
from .mixer_operating.mixer_operating import (
    MixerOperatingEnv,
    MixerOperatingTestEnv,
    MixerOperatingAgentEnv,
)

# high-level tasks
from .item_assembly.item_assembly import (
    ItemAssemblyEnv,
    ItemAssemblyAgentEnv,
)
from .manipulate_pipette.manipulate_pipette import (
    ManipulatePipetteEnv,
    ManipulatePipetteTestEnv,
    ManipulatePipetteAgentEnv,
)
from .sample_loading.sample_loading import (
    SampleLoadingEnv,
    SampleLoadingTestEnv,
    SampleLoadingAgentEnv,
)


# Competition-irrelevant tasks
# from ._other_tasks.open_pan.open_pan import (
#     OpenPanPickAndPlaceEnv,
#     OpenPanPickAndPlaceTestEnv,
#     OpenPanPickAndPlaceAgentEnv,
# )
# from ._other_tasks.sample_loading.sample_loading import (
#     SampleLoadingEnv,
#     SampleLoadingAgentEnv,
# )
# from ._other_tasks.manipulate_pipette_two_beaker.manipulate_pipette_two_beaker import (
#     ManipulatePipetteTwoBeakerEnv,
#     ManipulatePipetteTwoBeakerTestEnv,
#     ManipulatePipetteTwoBeakerAgentEnv,
# )
# from ._other_tasks.pour_water.pour_water import (
#     PourWaterEnv,
#     PourWaterAgentEnv,
# )