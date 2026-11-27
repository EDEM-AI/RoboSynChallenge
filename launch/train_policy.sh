#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 <act|diffusion|smolvla> <task|dataset_dir|Hub_ID> <output_dir> [lerobot-train options]"
    exit 0
fi
if [[ "$#" -lt 3 || ! "$1" =~ ^(act|diffusion|smolvla)$ ]]; then
    echo "Provide a policy, a dataset reference and an output directory." >&2
    exit 2
fi

policy=$1
dataset=$2
if [[ -d "$dataset" ]]; then
    dataset=$(realpath "$dataset")
fi
output=$(realpath -m "$3")
shift 3
cd "$REPO_ROOT"
exec uv run --locked --python 3.12 python -m scripts.train_lerobot "$policy" "$dataset" "$output" "$@"
