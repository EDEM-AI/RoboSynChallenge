#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

python launch/visualize_distribution/visualize_distribution.py \
    --env-name beaker_mixer_duel \
    --gym_config configs/beaker_mixer/gym_config_dual_clear.json \
    --action_config configs/beaker_mixer/action_config_dual.json \
    --resets 100 \
    --headless \
    "$@"
