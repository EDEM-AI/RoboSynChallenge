#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$SCRIPT_DIR/Rearrangement.sh" "$@"
"$SCRIPT_DIR/beaker_mixer_duel.sh" "$@"
"$SCRIPT_DIR/carry_basket.sh" "$@"
"$SCRIPT_DIR/click_button.sh" "$@"
"$SCRIPT_DIR/drawer_open_place.sh" "$@"
"$SCRIPT_DIR/items_handover_place.sh" "$@"
"$SCRIPT_DIR/manipulate_pipette_one_beaker.sh" "$@"
"$SCRIPT_DIR/open_pan.sh" "$@"
"$SCRIPT_DIR/pour_water_dual.sh" "$@"
"$SCRIPT_DIR/sample_loading_dual.sh" "$@"
