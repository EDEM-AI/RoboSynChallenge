#!/bin/bash
# ----------------------------------------------------------------------------
# bash finetune.sh <dataset_root> <output_dir> [gpu_id] [extra_opts...]
# bash finetune.sh /path/to/lerobot_v21_dataset outputs/train/act_click_bell 0 --steps 1000
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${3:-0}"

DATASET_ROOT="${1}"
OUTPUT_DIR="${2}"

shift 3 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$SCRIPT_DIR/training_data}"

echo "========================================="
echo "  ACT Policy Training"
echo "  Dataset: $DATASET_ROOT"
echo "  Output:  $OUTPUT_DIR"
echo "  GPU:     $GPU_ID"
echo "========================================="

cd "$SCRIPT_DIR"

uv run python scripts/train.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
