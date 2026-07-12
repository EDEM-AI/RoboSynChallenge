#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

POLICY_NAME=Your_Policy # [TODO]

TASK_NAME="${1}"
SETTING="${2}"
TRAIN_CONFIG="${3}"
MODEL_NAME="${4}"
GPU_ID="${5}"
# [TODO] add parameters here

shift 5 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4


echo "========================================="
echo "  Your-Own-Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  GPU:        $GPU_ID"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    exit 1
fi

cd "$REPO_ROOT" # move to RoboSynChallenge root

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --model_name "$MODEL_NAME" \
    # [TODO] add parameters here
