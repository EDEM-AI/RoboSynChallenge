#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../.." && pwd)
policy_name=$(basename "$script_dir")

task=${1:?Provide a task}
setting=${2:?Provide clear or random}
train_config=${3:?Provide a train config}
model=${4:?Provide a model}
gpu=${5:?Provide a GPU ID}
shift 5

export CUDA_VISIBLE_DEVICES="$gpu"
export EMBODICHAIN_SIM_EXIT_PROCESS=0
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-offscreen}"
    export SDL_VIDEO_DRIVER="${SDL_VIDEO_DRIVER:-offscreen}"
fi

cd "$repo"
exec uv run --locked --python 3.12 python -m scripts.eval_policy \
    --config "policy/$policy_name/deploy_policy.yml" --overrides \
    --task_name "$task" --setting "$setting" \
    --train_config_name "$train_config" --model_name "$model" "$@"
