#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ "$#" -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    echo -e "\n\033[1;33mUsage:\033[0m"
    echo -e "  $0 \033[1;32m<task_name>\033[0m \033[1;34m<clear_episodes>\033[0m \033[1;34m<random_episodes>\033[0m \033[1;34m<output_merge_path_or_id>\033[0m \033[1;35m[extra_args...]\033[0m\n"
    echo -e "This script automatically runs the clear and random settings for a task with specified episodes, finds the newly generated datasets, and merges them using \033[1;32mlerobot-edit-dataset\033[0m.\n"

    source "$SCRIPT_DIR/_print_available_tasks.sh"

    exit 0
fi

if [ "$#" -lt 4 ]; then
    echo -e "\n\033[1;31mError: Missing required arguments.\033[0m"
    echo -e "Run \033[1;35m$0 -h\033[0m for usage details.\n"
    exit 1
fi

TASK_NAME=$1
CLEAR_EPISODES=$2
RANDOM_EPISODES=$3
MERGED_OUTPUT=$4
shift 4
EXTRA_ARGS=("$@")

# Timestamp before running
START_TIME=$(date +%s)

echo "========================================="
echo -e "\033[1;36mStep 1: Generating Clear Dataset: max_episodes=$CLEAR_EPISODES\033[0m"
"$SCRIPT_DIR/run_task.sh" "$TASK_NAME" clear 2_1 --max_episodes "$CLEAR_EPISODES" "${EXTRA_ARGS[@]}"
sleep 5;

echo "========================================="
echo -e "\033[1;36mStep 2: Generating Random Dataset: max_episodes=$RANDOM_EPISODES\033[0m"
"$SCRIPT_DIR/run_task.sh" "$TASK_NAME" random 2_1 --max_episodes "$RANDOM_EPISODES" "${EXTRA_ARGS[@]}"
sleep 5;

echo "========================================="
echo -e "\033[1;36mStep 3: Locating newly generated datasets\033[0m"
echo "========================================="
# Assuming the datasets are saved either in lerobot_dataset/
LATEST_DATASETS=$(python scripts/_find_latest_datasets.py "$TASK_NAME" --count 2)

if [[ "$LATEST_DATASETS" == *"ERROR"* ]]; then
    echo -e "\033[1;31mError: Could not find two newly generated datasets to merge.\033[0m"
    exit 1
fi

DATASET_1=$(echo "$LATEST_DATASETS" | head -n 1)
DATASET_2=$(echo "$LATEST_DATASETS" | tail -n 1)

echo -e "Found Dataset 1 (Clear): \033[1;32m$DATASET_1\033[0m"
echo -e "Found Dataset 2 (Random): \033[1;32m$DATASET_2\033[0m"

echo "========================================="
echo -e "\033[1;36mStep 4: Merging datasets\033[0m"
echo "========================================="

# Attempt to use lerobot array parsing format
REPO_IDS="['$DATASET_1', '$DATASET_2']"
MERGED_OUTPUT="$TASK_NAME"/"$MERGED_OUTPUT"

export HF_LEROBOT_HOME=/root/workspace/RoboSynChallenge/lerobot_dataset/
echo -e "Running: \033[1;35mlerobot-edit-dataset --repo_id \"$MERGED_OUTPUT\" --push_to_hub false --operation.type merge --operation.repo_ids \"$REPO_IDS\"\033[0m"

lerobot-edit-dataset --repo_id "$MERGED_OUTPUT" --push_to_hub false --operation.type merge --operation.repo_ids "$REPO_IDS"

echo -e "\n\033[1;32mSuccessfully merged into: $MERGED_OUTPUT\033[0m"
chmod 777 -R "$REPO_ROOT/lerobot_dataset/"