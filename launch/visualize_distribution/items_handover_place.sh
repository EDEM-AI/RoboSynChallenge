#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

python launch/visualize_distribution/visualize_distribution.py \
    --env-name items_handover_place \
    --gym_config configs/items_handover_place/gym_config_clear.json \
    --action_config configs/items_handover_place/action_config.json \
    --resets 100 \
    --headless \
    "$@"
