#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
if [[ "$#" -lt 3 || ! "$1" =~ ^(act|dp|smolvla)$ ]]; then
    echo "Usage: $0 <act|dp|smolvla> <task> <clear|random> [checkpoint_dir|Hub_ID] [gpu_id] [overrides]" >&2
    exit 2
fi

policy=$1
task=$2
setting=$3
shift 3
checkpoint_options=()
if [[ "$#" -gt 0 && "$1" != --* && ( ! "$1" =~ ^[0-9]+$ || -d "$1" ) ]]; then
    checkpoint=$1
    if [[ -d "$checkpoint" ]]; then
        checkpoint=$(realpath "$checkpoint")
        if [[ -d "$checkpoint/pretrained_model" ]]; then
            checkpoint="$checkpoint/pretrained_model"
        fi
    fi
    checkpoint_options=(--checkpoint_path "$checkpoint" --model_name "$(basename "$checkpoint")")
    shift
else
    checkpoint_options=(--model_name "${policy}_sim_${task//\//_}")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="$1"
    shift
fi

cd "$REPO_ROOT"
export EMBODICHAIN_SIM_EXIT_PROCESS=0
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-offscreen}"
    export SDL_VIDEO_DRIVER="${SDL_VIDEO_DRIVER:-offscreen}"
fi
exec uv run --locked --python 3.12 python -m scripts.eval_policy \
    --config "policy/$policy/deploy_policy.yml" --overrides \
    --task_name "$task" --setting "$setting" "${checkpoint_options[@]}" "$@"
