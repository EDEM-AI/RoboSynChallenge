train_config_name=$1
model_name=$2
gpu_use=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_ROOT="$SCRIPT_DIR"

cd "$PI05_ROOT"

export CUDA_VISIBLE_DEVICES="$gpu_use"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export HF_LEROBOT_HOME="$PI05_ROOT/training_data"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py "$train_config_name" --exp-name="$model_name" --overwrite