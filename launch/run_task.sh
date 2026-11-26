#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 <task_name> <clear|random> [EmbodiChain run-task options]"
    echo "Datasets are recorded in LeRobot v3.0."
    echo "Use --seeded as the first option to collect from an explicit --seed."
    echo "Example: $0 click_bell clear --headless --max_episodes 1 --dataset-dir /path/to/data"
    exit 0
fi

if [[ "$#" -lt 2 || ! "$1" =~ ^(other_tasks/)?[a-z][a-z0-9_]*$ || ( "$2" != clear && "$2" != random ) ]]; then
    echo "Expected a task name and clear or random. Run $0 --help for usage." >&2
    exit 2
fi

TASK_CONFIG="$REPO_ROOT/configs/$1/$2/task.cobotmagic.yaml"
if [[ ! -f "$TASK_CONFIG" && "$1" != */* ]]; then
    TASK_CONFIG="$REPO_ROOT/configs/other_tasks/$1/$2/task.cobotmagic.yaml"
fi
shift 2
case "${1:-}" in
    3_0) shift ;;
    2_1) echo "Collection supports LeRobot v3.0." >&2; exit 2 ;;
esac
if [[ ! -f "$TASK_CONFIG" ]]; then
    echo "EmbodiChain 0.3.3 task deployment is missing: $TASK_CONFIG" >&2
    exit 1
fi

export EMBODICHAIN_DATA_ROOT="${EMBODICHAIN_DATA_ROOT:-$REPO_ROOT/assets}"
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-offscreen}"
    export SDL_VIDEO_DRIVER="${SDL_VIDEO_DRIVER:-offscreen}"
fi
cd "$REPO_ROOT"
if [[ "${1:-}" == --seeded ]]; then
    shift
    export EMBODICHAIN_SIM_EXIT_PROCESS=0
    exec uv run --locked --python 3.12 python -m scripts.collect_seeded \
        --gym_config "$TASK_CONFIG" --num_envs 1 "$@"
fi
for option in "$@"; do
    if [[ "$option" == --replay ]]; then
        exec uv run --locked --python 3.12 python -m scripts.run_env \
            --gym_config "$TASK_CONFIG" --num_envs 1 "$@"
    fi
done
exec uv run --locked --python 3.12 embodichain run-task \
    --gym_config "$TASK_CONFIG" --num_envs 1 "$@"
