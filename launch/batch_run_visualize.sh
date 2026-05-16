#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Usage: ./batch_run_visualize.sh [setting(random|clear)] [extra_args...]
# Default setting is 'clear' if not provided
SETTING=${1:-clear}

if [[ "$#" -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    echo -e "\n\033[1;33mUsage:\033[0m"
    echo -e "  $0 \033[1;34m[setting(random|clear)]\033[0m \033[1;35m[extra_args...]\033[0m\n"
    exit 0
fi

# Remove the setting argument if provided so we can pass extra args
if [ "$#" -ge 1 ]; then
    shift 1
fi
EXTRA_ARGS=("$@")

# Define the list of tasks to run
TASKS=(
    "click_bell"
    "handle_basket"
    "water_pouring"
    "table_rearrangement"
    "items_handover"
    "drawer_open_place"
    "mixer_operating"
    "item_assembly"
    "manipulate_pipette"
    "sample_loading"
)

echo "========================================="
echo -e "\033[1;36mBatch Running Visualizations\033[0m"
echo -e "Setting: \033[1;34m$SETTING\033[0m"
echo "Total Tasks: ${#TASKS[@]}"
echo "========================================="

# Loop through all tasks and run them using the new unified script
for TASK in "${TASKS[@]}"; do
    echo -e "\n\033[1;32m>>> Starting task: $TASK <<<\033[0m"
    "$SCRIPT_DIR/run_visualize.sh" "$TASK" "$SETTING" "${EXTRA_ARGS[@]}"
done

echo -e "\n\033[1;32mAll batch visualization tasks completed!\033[0m"
