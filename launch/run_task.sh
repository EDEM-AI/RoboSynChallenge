#!/bin/bash

# Usage: ./run_task.sh <task_name> [random | clear] [extra_args...]
# Examples:
#   ./run_task.sh beaker_mixer_duel random
#   ./run_task.sh pour_water_dual clear
# In addition to the preset parameters in the script,
# you can also input the following additional parameters supported by embodichain:
# --filter_visual_rand: to disable visual randomization
# --filter_dataset_saving: to disable dataset saving

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ "$#" -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    echo -e "\n\033[1;33mUsage:\033[0m"
    echo -e "  $0 \033[1;32m<task_name>\033[0m \033[1;34m<setting(random|clear)>\033[0m \033[1;34m<format(3_0|2_1)>\033[0m \033[1;35m[extra_args...]\033[0m\n"

    echo -e "\033[1;33mAvailable Extra Arguments:\033[0m"
    echo -e "  \033[1;35m--filter_visual_rand\033[0m     : Disable visual randomization"
    echo -e "  \033[1;35m--filter_dataset_saving\033[0m  : Disable dataset saving"
    echo -e "  \033[1;35m--max_episodes <num>\033[0m  : Specify the maximum number of episodes to generate"
    echo -e "  \033[1;35m--headless\033[0m  : Run in headless mode\n"

    source "$SCRIPT_DIR/_print_available_tasks.sh"

    exit 0
fi

if [ "$#" -lt 3 ]; then
    echo -e "\n\033[1;31mError: Missing required arguments.\033[0m"
    echo -e "Run \033[1;35m$0 -h\033[0m or \033[1;35m$0 --help\033[0m for usage details.\n"
    exit 1
fi

TASK_NAME=$1
SETTING=$2
FORMAT=$3
shift 3

EXTRA_ARGS=("$@")

# Dynamically combine paths
# Based on the new structure, gym_config is located in the random or clear folder under the task name
GYM_CONFIG="configs/${TASK_NAME}/${SETTING}/gym_config.json"

# action_config is usually in the task root directory, or under the corresponding setting folder
if [ -f "configs/${TASK_NAME}/action_config.json" ]; then
    ACTION_CONFIG="configs/${TASK_NAME}/action_config.json"
else
    # Fallback to check if it's placed in the random/clear folder
    ACTION_CONFIG="configs/${TASK_NAME}/${SETTING}/action_config.json"
fi

# Check if files exist
if [ ! -f "$GYM_CONFIG" ]; then
    echo "Error: Cannot find corresponding gym_config: $GYM_CONFIG"
    exit 1
fi

if [ ! -f "$ACTION_CONFIG" ]; then
    echo "Error: Cannot find corresponding action_config: $ACTION_CONFIG"
    exit 1
fi

echo "========================================="
echo "Executing task: $TASK_NAME ($SETTING)"
echo "GYM_CONFIG: $GYM_CONFIG"
echo "ACTION_CONFIG: $ACTION_CONFIG"
echo "========================================="

# Execute the Python script, keeping default parameters from the original scripts
# Extract original extra arguments, such as --filter_visual_rand
RUN_CMD=(
    python -m scripts.run_env
    --gym_config "$GYM_CONFIG"
    --action_config "$ACTION_CONFIG"
    --num_envs 1
)

RUN_CMD+=("${EXTRA_ARGS[@]}")

echo "Running command:"
echo "${RUN_CMD[@]}"
echo "========================================="

"${RUN_CMD[@]}"
sleep 5;

if [ "$FORMAT" == "2_1" ]; then
    echo "========================================="
    echo -e "\033[1;36mConverting newly generated dataset to 2.1 format...\033[0m"

    LATEST_REL_PATH=$(python scripts/_find_latest_datasets.py "$TASK_NAME" --count 1)

    if [[ "$LATEST_REL_PATH" == *"ERROR"* ]]; then
        echo -e "\033[1;31mError: Could not find generated dataset for conversion.\033[0m"
    else
        DATASET_ID=$(basename "$LATEST_REL_PATH")
        DATASET_ROOT="$REPO_ROOT/lerobot_dataset/$TASK_NAME"

        echo -e "Found New Dataset: \033[1;32m$DATASET_ID\033[0m"

        echo "Running conversion: scripts/convert_lerobot3.0_to_2.1.py --repo-id \"$DATASET_ID\" --root \"$DATASET_ROOT\""
        python scripts/convert_lerobot3.0_to_2.1.py --repo-id "$DATASET_ID" --root "$DATASET_ROOT"

        echo -e "\033[1;32mConversion completed for: $DATASET_ID\033[0m"

        if [ -d "$REPO_ROOT/lerobot_dataset/${LATEST_REL_PATH}_v3.0" ]; then
            echo -e "\033[1;32mOriginal 3.0 dataset removed: $REPO_ROOT/lerobot_dataset/${LATEST_REL_PATH}_v3.0\033[0m"
            rm -rf "$REPO_ROOT/lerobot_dataset/${LATEST_REL_PATH}_v3.0"
        fi
    fi
fi
chmod 777 -R "$REPO_ROOT/lerobot_dataset/"

