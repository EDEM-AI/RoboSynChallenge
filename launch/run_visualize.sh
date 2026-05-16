#!/bin/bash
set -euo pipefail

# Usage: ./run_visualize.sh <task_name> <setting(random|clear)> [extra_args...]
# Examples:
#   ./run_visualize.sh click_bell random
#   ./run_visualize.sh manipulate_pipette clear

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ "$#" -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    echo -e "\n\033[1;33mUsage:\033[0m"
    echo -e "  $0 \033[1;32m<task_name>\033[0m \033[1;34m<setting(random|clear)>\033[0m \033[1;35m[extra_args...]\033[0m\n"

    echo -e "\033[1;33mAvailable Extra Arguments (Examples):\033[0m"
    echo -e "  \033[1;35m--resets\033[0m       : Number of resets to visualize (default: 100)"
    echo -e "  \033[1;35m--headless\033[0m     : Run in headless mode (default: true)\n"

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
GYM_CONFIG="configs/${TASK_NAME}/${SETTING}/gym_config.json"

if [ -f "configs/${TASK_NAME}/action_config.json" ]; then
    ACTION_CONFIG="configs/${TASK_NAME}/action_config.json"
else
    ACTION_CONFIG="configs/${TASK_NAME}/${SETTING}/action_config.json"
fi

if [ ! -f "$GYM_CONFIG" ]; then
    echo -e "\033[1;31mError: Cannot find corresponding gym_config: $GYM_CONFIG\033[0m"
    exit 1
fi

if [ ! -f "$ACTION_CONFIG" ]; then
    echo -e "\033[1;31mError: Cannot find corresponding action_config: $ACTION_CONFIG\033[0m"
    exit 1
fi

echo "========================================="
echo -e "Visualizing task: \033[1;32m$TASK_NAME\033[0m (\033[1;34m$SETTING\033[0m)"
echo "GYM_CONFIG: $GYM_CONFIG"
echo "ACTION_CONFIG: $ACTION_CONFIG"
echo "========================================="

RUN_CMD=(
    python scripts/visualize_distribution.py
    --env-name "$TASK_NAME"
    --gym_config "$GYM_CONFIG"
    --action_config "$ACTION_CONFIG"
    --resets 100
    --headless
)

RUN_CMD+=("${EXTRA_ARGS[@]}")

echo "Running command:"
echo "${RUN_CMD[@]}"
"${RUN_CMD[@]}"