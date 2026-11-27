#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 <task_name> <clear_episodes> <random_episodes> <output_dir> [run-task options]"
    echo "Collects and merges LeRobot v3.0 data."
    exit 0
fi
if [[ "$#" -lt 4 || ! "$1" =~ ^(other_tasks/)?[a-z][a-z0-9_]*$ || ! "$2" =~ ^[1-9][0-9]*$ || ! "$3" =~ ^[1-9][0-9]*$ ]]; then
    echo "Provide a task, two positive episode counts and an output directory." >&2
    exit 2
fi

task=$1
clear_episodes=$2
random_episodes=$3
case "$4" in
    */*) output=$(realpath -m "$4") ;;
    *) output="$REPO_ROOT/lerobot_dataset/$task/$4" ;;
esac
shift 4
if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    exit 1
fi

mkdir -p "$(dirname "$output")"
label=${task//\//-}
sources=$(mktemp -d "$(dirname "$output")/.collect-$label.XXXXXX")
echo "Source recordings: $sources"
"$SCRIPT_DIR/run_task.sh" "$task" clear "$@" --max_episodes "$clear_episodes" --dataset-dir "$sources/clear"
"$SCRIPT_DIR/run_task.sh" "$task" random "$@" --max_episodes "$random_episodes" --dataset-dir "$sources/random"

shopt -s nullglob
clear_info=("$sources"/clear/*/meta/info.json)
random_info=("$sources"/random/*/meta/info.json)
if [[ "${#clear_info[@]}" -ne 1 || "${#random_info[@]}" -ne 1 ]]; then
    echo "Expected one finalized recorder dataset from each scene in $sources." >&2
    exit 1
fi

cd "$REPO_ROOT"
exec uv run --locked --python 3.12 python -m scripts.merge_lerobot_v30 \
    --repo-id "local/$label" --output "$output" \
    "${clear_info[0]%/meta/info.json}" "${random_info[0]%/meta/info.json}"
