#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random /path/to/checkpoint_or_pretrained_model 5 \
#   --pytorch_device cuda --headless true --eval_video_log true
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
if [[ -z "${EMBODICHAIN_ROOT:-}" ]]; then
    if [[ -d "$WORKSPACE_ROOT/EmbodiChain" ]]; then
        EMBODICHAIN_ROOT="$WORKSPACE_ROOT/EmbodiChain"
    elif [[ -d /data1/jinruixing/RoboSynChallenge_ws_new/EmbodiChain ]]; then
        EMBODICHAIN_ROOT=/data1/jinruixing/RoboSynChallenge_ws_new/EmbodiChain
    else
        EMBODICHAIN_ROOT="$WORKSPACE_ROOT/EmbodiChain"
    fi
fi
VENV_DIR="${ROBOSYN_VENV_DIR:-$EMBODICHAIN_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

POLICY_NAME=smolvla

TASK_NAME="${1:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
SETTING="${2:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
CHECKPOINT_PATH="${3:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

if [[ -n "${SMOLVLA_PYTHON:-}" ]]; then
    EXTRA_ARGS+=(--smolvla_python "$SMOLVLA_PYTHON")
elif [[ -x /home/jinruixing/miniconda3/envs/smolvla/bin/python ]]; then
    EXTRA_ARGS+=(--smolvla_python /home/jinruixing/miniconda3/envs/smolvla/bin/python)
fi
if [[ -n "${SMOLVLA_LEROBOT_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--lerobot_root "$SMOLVLA_LEROBOT_ROOT")
elif [[ -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    EXTRA_ARGS+=(--lerobot_root "$SCRIPT_DIR/lerobot")
elif [[ -d /data1/jinruixing/lerobot/src/lerobot ]]; then
    EXTRA_ARGS+=(--lerobot_root /data1/jinruixing/lerobot)
fi

if [[ -d "$CHECKPOINT_PATH/pretrained_model" ]]; then
    CHECKPOINT_DIR="$CHECKPOINT_PATH/pretrained_model"
else
    CHECKPOINT_DIR="$CHECKPOINT_PATH"
fi

# DexSim uses physical GPU ordinals from --gpu_id. Keep the simulator parent
# unmasked, then mask only the SmolVLA worker via --smolvla_cuda_visible_devices.
unset CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data1/jinruixing/.cache/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  SmolVLA Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  GPU:        $GPU_ID"
echo "  Python:     $PYTHON_BIN"
if [[ -n "${SMOLVLA_PYTHON:-}" ]]; then
    echo "  Worker Py:  $SMOLVLA_PYTHON"
elif [[ -x /home/jinruixing/miniconda3/envs/smolvla/bin/python ]]; then
    echo "  Worker Py:  /home/jinruixing/miniconda3/envs/smolvla/bin/python"
fi
if [[ -n "${SMOLVLA_LEROBOT_ROOT:-}" ]]; then
    echo "  LeRobot:    $SMOLVLA_LEROBOT_ROOT"
elif [[ -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    echo "  LeRobot:    $SCRIPT_DIR/lerobot"
elif [[ -d /data1/jinruixing/lerobot/src/lerobot ]]; then
    echo "  LeRobot:    /data1/jinruixing/lerobot"
else
    echo "  LeRobot:    installed package in worker Python"
fi
echo "========================================="

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: cannot find executable Python: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_name "$(basename "$(dirname "$CHECKPOINT_DIR")")" \
    --gpu_id "$GPU_ID" \
    --smolvla_cuda_visible_devices "$GPU_ID" \
    "${EXTRA_ARGS[@]}"
