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

from .dummy_task import DummyTaskEnv
from .simple_motion import SimpleMotionEnv

from .items_handover_place.items_handover_place import (
    ItemsHandoverPlaceEnv,
)
from .sample_loading.sample_loading import (
    SampleLoadingEnv,
)
from .sample_loading_duel.sample_loading_duel import (
    SampleLoadingDuelEnv,
)
from .manipulate_pipette.manipulate_pipette import (
    ManipulatePipetteEnv,
)
from .drawer_open_place.drawer_open_place import (
    DrawerOpenPlaceAgentEnv,
    DrawerOpenPlaceEnv,
)
from .beaker_mixer.beaker_mixer import (
    BeakerMixerEnv,
)
from .pour_water_dual.pour_water import (
    PourWaterDualEnv,
    PourWaterDualAgentEnv,
)
from .carry_basket.carry_basket import (
    CarryBasketEnvV2,
    CarryBasketAgentEnvV2,
)
from .Beat_Hammer_Block.Beat_Hammer_Block import (
    BeatHammerBlockEnv,
    BeatHammerBlockAgentEnv,
)

from .open_pan.open_pan import (
    OpenPanPickAndPlaceEnv,
    OpenPanPickAndPlaceAgentEnv,
)
from .rearrangement import (
    RearrangementEnv,
    RearrangementAgentEnv,
)
from .Articulated_object_operation.Articulated_object_operation import (
    ArticulatedobjectoperationEnv,
    ArticulatedobjectoperationAgentEnv,
)