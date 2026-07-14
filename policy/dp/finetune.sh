#!/bin/bash
# ----------------------------------------------------------------------------
# bash finetune.sh <dataset_root> <output_dir> [gpu_id] [extra_opts...]
# bash finetune.sh /path/to/lerobot_v21_dataset outputs/train/dp_click_bell 0 --steps 1000
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${3:-0}"

DATASET_ROOT="${1}"
OUTPUT_DIR="${2}"

shift 3 2>/dev/null || true
EXTRA_ARGS=("$@")

# CUDA_VISIBLE_DEVICES controls which GPU(s) this single-process launcher sees.
# For DDP/multi-GPU training, prefer the torchrun command in the DP docs.
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# HF_LEROBOT_HOME is where LeRobot resolves local datasets and metadata.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$SCRIPT_DIR/training_data}"

echo "========================================="
echo "  Diffusion Policy Training"
echo "  Dataset: $DATASET_ROOT"
echo "  Output:  $OUTPUT_DIR"
echo "  GPU:     $GPU_ID"
echo "========================================="

cd "$SCRIPT_DIR"

uv run python scripts/train.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
