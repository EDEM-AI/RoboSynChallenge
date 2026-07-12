#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random /path/to/checkpoint 0 --max_episodes 5
# ----------------------------------------------------------------------------

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

POLICY_NAME=act

TASK_NAME="${1}"
SETTING="${2}"
CHECKPOINT_PATH="${3}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"


echo "========================================="
echo "  ACT Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  GPU:        $GPU_ID"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
cd "$REPO_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model_name "$(basename "$CHECKPOINT_PATH")" \
    "${EXTRA_ARGS[@]}"
