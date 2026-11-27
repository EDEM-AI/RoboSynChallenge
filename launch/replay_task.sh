#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 <task_name> <clear|random> <trajectory.pt> [EmbodiChain replay options]"
    echo "Example: $0 click_bell clear trajectory.pt --headless --replay_mode dynamic"
    echo "Use --seeded-reset as the first replay option to start from the original seed."
    echo "Use --recorded-reset to restore the initial state saved in the trajectory."
    exit 0
fi
if [[ "$#" -lt 3 || ! -f "$3" ]]; then
    echo "Expected a task name, scene setting and existing trajectory. Run $0 --help for usage." >&2
    exit 2
fi

task=$1
setting=$2
trajectory=$(realpath "$3")
shift 3
if [[ "${1:-}" == --seeded-reset || "${1:-}" == --recorded-reset ]]; then
    reset_options=()
    if [[ "$1" == --recorded-reset ]]; then
        reset_options+=(--recorded-reset)
    fi
    shift
    if [[ ! "$task" =~ ^(other_tasks/)?[a-z][a-z0-9_]*$ || ( "$setting" != clear && "$setting" != random ) ]]; then
        echo "Expected a task name and clear or random. Run $0 --help for usage." >&2
        exit 2
    fi
    REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
    export EMBODICHAIN_DATA_ROOT="${EMBODICHAIN_DATA_ROOT:-$REPO_ROOT/assets}"
    export EMBODICHAIN_SIM_EXIT_PROCESS=0
    if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-offscreen}"
        export SDL_VIDEO_DRIVER="${SDL_VIDEO_DRIVER:-offscreen}"
    fi
    TASK_CONFIG="$REPO_ROOT/configs/$task/$setting/task.cobotmagic.yaml"
    if [[ ! -f "$TASK_CONFIG" && "$task" != */* ]]; then
        TASK_CONFIG="$REPO_ROOT/configs/other_tasks/$task/$setting/task.cobotmagic.yaml"
    fi
    cd "$REPO_ROOT"
    exec uv run --locked --python 3.12 python -m scripts.replay_seeded \
        --gym_config "$TASK_CONFIG" \
        "$trajectory" "${reset_options[@]}" "$@"
fi
exec "$SCRIPT_DIR/run_task.sh" "$task" "$setting" \
    --filter_dataset_saving --replay --replay_trajectory "$trajectory" "$@"
