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
    echo -e "  $0 \033[1;32m<task_name>\033[0m \033[1;34m<setting(random|clear)>\033[0m \033[1;35m[extra_args...]\033[0m\n"

    echo -e "\033[1;33mAvailable Extra Arguments:\033[0m"
    echo -e "  \033[1;35m--filter_visual_rand\033[0m     : Disable visual randomization"
    echo -e "  \033[1;35m--filter_dataset_saving\033[0m  : Disable dataset saving\n"

    echo -e "\033[1;33mAvailable task name examples:\033[0m"

    echo -e "  \033[1;36m[ Low-level tasks ]\033[0m"
    echo -e "    \033[0;32m✦ click_bell\033[0m"
    echo -e "    \033[0;32m✦ handle_basket\033[0m"
    echo -e "    \033[0;32m✦ water_pouring\033[0m"
    echo -e "    \033[0;32m✦ table_rearrangement\033[0m\n"

    echo -e "  \033[1;36m[ Mid-level tasks ]\033[0m"
    echo -e "    \033[0;32m✦ items_handover\033[0m"
    echo -e "    \033[0;32m✦ drawer_open_place\033[0m"
    echo -e "    \033[0;32m✦ mixer_operating\033[0m\n"

    echo -e "  \033[1;36m[ High-level tasks ]\033[0m"
    echo -e "    \033[0;32m✦ item_assembly\033[0m"
    echo -e "    \033[0;32m✦ manipulate_pipette\033[0m"
    echo -e "    \033[0;32m✦ sample_loading\033[0m\n"
    exit 0
fi

if [ "$#" -lt 2 ]; then
    echo -e "\n\033[1;31mError: Missing required arguments.\033[0m"
    echo -e "Run \033[1;35m$0 -h\033[0m or \033[1;35m$0 --help\033[0m for usage details.\n"
    exit 1
fi

TASK_NAME=$1
SETTING=$2
shift 2
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
    --enable_rt
)

RUN_CMD+=("${EXTRA_ARGS[@]}")

echo "Running command:"
echo "${RUN_CMD[@]}"
"${RUN_CMD[@]}"
