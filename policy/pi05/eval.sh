#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <train_config> <model_name> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random pi0_base_robosynchallenge_full pi0 0 --max_episodes 50
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python}"

POLICY_NAME=pi05

TASK_NAME="${1}"
SETTING="${2}"
TRAIN_CONFIG="${3}"
MODEL_NAME="${4}"
GPU_ID="${5}"

shift 5 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4


echo "========================================="
echo "  π0.5 Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Config:     $TRAIN_CONFIG / $MODEL_NAME"
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
    --model_name "$MODEL_NAME" \
    --train_config_name "$TRAIN_CONFIG" \
    "${EXTRA_ARGS[@]}"
