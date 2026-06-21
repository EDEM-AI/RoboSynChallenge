train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES

export HF_LEROBOT_HOME=/root/workspace/RoboSynChallenge/policy/pi0/training_data/
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite