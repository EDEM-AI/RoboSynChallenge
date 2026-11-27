#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset=${1:?Usage: bash finetune.sh <task|dataset_dir|Hub_ID> <output_dir> [gpu_id] [lerobot-train options]}
output=${2:?Specify an output directory}
shift 2
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="$1"
    shift
fi

exec "$SCRIPT_DIR/../../launch/train_policy.sh" diffusion "$dataset" "$output" \
    --batch_size=8 --num_workers=4 --steps=100000 --save_freq=20000 --log_freq=200 \
    --env_eval_freq=0 --seed=1000 --policy.horizon=16 --policy.n_action_steps=8 "$@"
