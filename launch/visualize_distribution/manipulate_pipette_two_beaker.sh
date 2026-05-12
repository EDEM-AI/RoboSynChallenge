#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

python launch/visualize_distribution/visualize_distribution.py \
    --env-name manipulate_pipette_two_beaker \
    --gym_config configs/manipulate_pipette/gym_config_two_beaker_clear.json \
    --action_config configs/manipulate_pipette/action_config_two_beaker.json \
    --resets 100 \
    --headless \
    "$@"
