#!/bin/bash
set -euo pipefail

# Usage: ./replay_task.sh <task_name> <setting(random|clear)> <trajectory.pt> [extra_args...]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ "$#" -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    echo "Usage:"
    echo "  $0 <task_name> <setting(random|clear)> <trajectory.pt> [extra_args...]"
    echo
    echo "Examples:"
    echo "  $0 drawer_open_place random /path/to/state_action.pt --replay_mode dynamic"
    echo "  $0 drawer_open_place clear /path/to/trajectory.pt --replay_mode kinematic"
    exit 0
fi

if [ "$#" -lt 3 ]; then
    echo "Error: task name, setting, and trajectory path are required." >&2
    echo "Run $0 --help for usage." >&2
    exit 1
fi

TASK_NAME=$1
SETTING=$2
TRAJECTORY=$3
shift 3

GYM_CONFIG="configs/${TASK_NAME}/${SETTING}/gym_config.json"
if [ -f "configs/${TASK_NAME}/action_config.json" ]; then
    ACTION_CONFIG="configs/${TASK_NAME}/action_config.json"
else
    ACTION_CONFIG="configs/${TASK_NAME}/${SETTING}/action_config.json"
fi

if [ ! -f "$GYM_CONFIG" ]; then
    echo "Error: Cannot find gym config: $GYM_CONFIG" >&2
    exit 1
fi
if [ ! -f "$ACTION_CONFIG" ]; then
    echo "Error: Cannot find action config: $ACTION_CONFIG" >&2
    exit 1
fi
if [ ! -f "$TRAJECTORY" ]; then
    echo "Error: Cannot find trajectory: $TRAJECTORY" >&2
    exit 1
fi

RUN_CMD=(
    python -m scripts.run_env
    --gym_config "$GYM_CONFIG"
    --action_config "$ACTION_CONFIG"
    --num_envs 1
    --filter_dataset_saving
    --replay
    --replay_trajectory "$TRAJECTORY"
)
RUN_CMD+=("$@")

echo "Replaying task: $TASK_NAME ($SETTING)"
echo "Trajectory: $TRAJECTORY"
echo "Running command:"
echo "${RUN_CMD[@]}"
"${RUN_CMD[@]}"
