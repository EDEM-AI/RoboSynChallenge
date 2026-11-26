#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/.." && pwd)
if [[ "$#" -lt 2 || ! "$1" =~ ^(pi0|pi05)$ ]]; then
    echo "Usage: $0 <pi0|pi05> <train|stats|eval|serve> [arguments]" >&2
    exit 2
fi
policy=$1
mode=$2
shift 2
source_dir="$repo/policy/$policy"
export PYTHONPATH="$repo:$source_dir/src:$source_dir/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

case "$mode" in
    train)
        config=${1:?Provide an OpenPI training config}
        name=${2:?Provide an experiment name}
        gpu=${3:?Provide a GPU ID}
        shift 3
        export CUDA_VISIBLE_DEVICES="$gpu"
        cd "$source_dir"
        exec uv run --project "$repo" --locked --python 3.12 python scripts/train.py "$config" --exp-name="$name" "$@"
        ;;
    stats)
        cd "$source_dir"
        exec uv run --project "$repo" --locked --python 3.12 python scripts/compute_norm_stats.py "$@"
        ;;
    serve)
        if [[ "$#" -eq 0 && -n "${SERVER_ARGS:-}" ]]; then
            read -r -a server_options <<< "$SERVER_ARGS"
            set -- "${server_options[@]}"
        fi
        cd "$source_dir"
        exec uv run --project "$repo" --locked --python 3.12 python scripts/serve_policy.py "$@"
        ;;
    eval)
        task=${1:?Provide a task}
        setting=${2:?Provide clear or random}
        config=${3:?Provide an OpenPI training config}
        name=${4:?Provide an experiment name}
        gpu=${5:?Provide a GPU ID}
        shift 5
        export CUDA_VISIBLE_DEVICES="$gpu"
        export EMBODICHAIN_SIM_EXIT_PROCESS=0
        if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
            export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-offscreen}"
            export SDL_VIDEO_DRIVER="${SDL_VIDEO_DRIVER:-offscreen}"
        fi
        cd "$repo"
        exec uv run --project "$repo" --locked --python 3.12 python -m scripts.eval_policy \
            --config "policy/$policy/deploy_policy.yml" --overrides \
            --task_name "$task" --setting "$setting" --train_config_name "$config" --model_name "$name" "$@"
        ;;
    *)
        echo "Unsupported OpenPI operation $mode." >&2
        exit 2
        ;;
esac
