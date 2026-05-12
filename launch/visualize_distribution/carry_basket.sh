#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

python launch/visualize_distribution/visualize_distribution.py \
    --env-name carry_basket \
    --gym_config configs/carry_basket/gym_config_clear.json \
    --action_config configs/carry_basket/action_config.json \
    --resets 100 \
    --headless \
    "$@"
