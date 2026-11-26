#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset=${1:?Provide a task, LeRobot v3.0 dataset directory, or Hub dataset ID.}
output=${2:?Provide an output directory.}
shift 2
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="$1"
    shift
fi

exec "$script_dir/../../launch/train_policy.sh" smolvla "$dataset" "$output" "$@"
