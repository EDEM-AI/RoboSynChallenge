#!/bin/bash
# ----------------------------------------------------------------------------
# eval.sh — 使用 π₀ 策略评估 RoboSynChallenge 任务
#
# 用法:
#   ./policy/pi0/eval.sh <task_name> <setting> <train_config> <model_name> [checkpoint_id] [gpu_id] [extra_opts...]
#
# 示例:
#   ./policy/pi0/eval.sh click_bell random my_config pi0_base 30000 0
#   ./policy/pi0/eval.sh water_pouring clear wpm2_embodichain pi0_wpm2 10000 1 --max_episodes 50
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python}"

POLICY_NAME=pi0

TASK_NAME="${1}"
SETTING="${2}"
TRAIN_CONFIG="${3}"
MODEL_NAME="${4}"
CHECKPOINT_ID="${5}"
GPU_ID="${6}"

shift 6 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4


echo "========================================="
echo "  π₀ Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Config:     $TRAIN_CONFIG / $MODEL_NAME @ $CHECKPOINT_ID"
echo "  GPU:        $GPU_ID"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    exit 1
fi

export PI0_VENV_DIR="$VENV_DIR"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$SCRIPT_DIR/src:$SCRIPT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
cd "$REPO_ROOT" # move to RoboSynChallenge root

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --train_config_name "$TRAIN_CONFIG" \
    --model_name "$MODEL_NAME" \
    --checkpoint_id "$CHECKPOINT_ID" \
    "${EXTRA_ARGS[@]}"
