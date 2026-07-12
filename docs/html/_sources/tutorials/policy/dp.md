# DP
## Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies.
```bash
conda activate robosyn
# Install uv
pip install uv
```

Once uv is installed, run the following commands to set up the environment:
```bash
cd policy/dp
uv sync --frozen
source .venv/bin/activate
```

The DP environment is defined by `policy/dp/pyproject.toml` and `policy/dp/uv.lock`. If you are running on a machine without network access during training or evaluation, activate the environment first and use `uv run --locked --no-sync --python .venv/bin/python ...`.

## Generate RoboSynChallenge Data
See <a href="../collect_data.html">Collect Data Section</a> for more details.

## Prepare DP Data for Training
DP training reads a RoboSynChallenge LeRobot dataset directly from `--dataset-root`. The wrapper in `policy/dp/scripts/train.py` maps the RoboSynChallenge dataset feature names to the LeRobot Diffusion Policy feature names expected by the installed LeRobot policy.

For a single task, pass the task dataset directory directly:
```shell
/path/to/RoboSynChallenge/lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell
```

If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), you can also use the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) to merge datasets. Here, we provide an example of using lerobot-edit-dataset to merge datasets:

Assume the two dataset directories are `/root/workspace/RoboSynChallenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Sim_beaker_mixer_dual` and `/root/workspace/RoboSynChallenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Real_beaker_mixer_dual`, you can use the following script and configuration file to merge it into `cobotmagic_merge_beaker_mixer_dual` in the same dir.
First, you can create a merge_config.json
```
{
  "repo_id": "lerobot_dataset/cobotmagic_merge_beaker_mixer_dual",
  "push_to_hub": false,
  "operation": {
    "type": "merge",
    "repo_ids": [
      "lerobot_dataset/cobotmagic_Sim_beaker_mixer_dual",
      "lerobot_dataset/cobotmagic_Real_beaker_mixer_dual"
    ]
  }
}
```
Then, use the following code:
```shell
export HF_LEROBOT_HOME=/root/workspace/RoboSynChallenge/
lerobot-edit-dataset --config_path /root/workspace/RoboSynChallenge/merge_config.json
```

After preparing the data, keep the dataset directory available and pass it to `finetune.sh`. You do not need to copy the data into `policy/dp`.

## Write the Corresponding `train_config`
DP uses the LeRobot `DiffusionConfig` assembled by `policy/dp/scripts/train.py`. You usually do not need to edit source code for a new task. Instead, pass the dataset path, output path, and training arguments from the command line.

Common task-specific arguments are:
```
--batch-size ${batch_size}
--horizon ${horizon}
--n-action-steps ${n_action_steps}
--num-inference-steps ${num_inference_steps}
--crop-shape ${height} ${width}
--img-micro-bs ${img_micro_bs}
--steps ${steps}
--log-freq ${log_freq}
--save-freq ${save_freq}
```

For multi-GPU training, launch `scripts/train.py` with `torchrun`, pass `--distributed`, and set the per-process `--batch-size`. For example, global batch size 64 on 2 GPUs uses `--batch-size 32`.

## Finetune model
```bash
# dataset_root: path to the RoboSynChallenge LeRobot dataset
# output_dir: where checkpoints will be saved
# gpu_use: if not using multi gpu, set to gpu_id like 0; else set like 0,1
bash policy/dp/finetune.sh ${dataset_root} ${output_dir} ${gpu_use} \
  --batch-size 64 \
  --horizon 32 \
  --n-action-steps 32 \
  --img-micro-bs 64 \
  --log-freq 100 \
  --save-freq 10000 \
  --wandb \
  --overwrite
```

For 2-GPU training with global batch size 64:
```bash
cd policy/dp
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --distributed \
  --dataset-root ${dataset_root} \
  --output-dir ${output_dir} \
  --batch-size 32 \
  --horizon 32 \
  --n-action-steps 32 \
  --img-micro-bs 64 \
  --steps 100000 \
  --log-freq 100 \
  --save-freq 10000 \
  --wandb \
  --wandb-project robosynchallenge \
  --wandb-name ${run_name} \
  --overwrite
```

| Training mode | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Single GPU DP | > 24 GB | RTX 5090 |
| 2-GPU DP | > 24 GB per GPU | 2\*RTX 5090 |

If your GPU memory is insufficient, reduce `--batch-size`, keep `--img-micro-bs 64`, reduce `--horizon`, or enable multi-GPU training with a smaller per-GPU batch size. The `--crop-shape` option changes the image crop before the DP RGB encoder; omit it to train with the full three-view observation.

The default `batch_size` in `scripts/train.py` is 8, but the recommended RoboSynChallenge DP training command uses global batch size 64 with horizon 32.

| Global batch size | GPU num | Per-GPU `--batch-size` | Example GPU |
| ----- | ----- | ----- | ----- |
| 64 | 1 | 64 | RTX 5090 |
| 64 | 2 | 32 | 2\*RTX 5090 |
| 64 | 4 | 16 | 4\*RTX 5090 |

## Eval on RoboSynChallenge

Checkpoints will be saved in `${output_dir}/checkpoints/${checkpoint_id}/pretrained_model` for single-GPU training. In distributed training, every rank writes a full checkpoint under `${output_dir}/rank_${rank}/checkpoints/${checkpoint_id}/pretrained_model`; use `rank_0` for evaluation and release.

```
# checkpoint_path like: checkpoints/dp_click_bell_h32_a32_b64_ddp2/rank_0/checkpoints/100000/pretrained_model
PYTHON_BIN=policy/dp/.venv/bin/python bash policy/dp/eval.sh ${task_name} [random | clear | random_eval_once] ${checkpoint_path} ${gpu_id} \
  --pytorch_device cuda \
  --headless True
# PYTHON_BIN=policy/dp/.venv/bin/python bash policy/dp/eval.sh click_bell random_eval_once checkpoints/dp_click_bell_h32_a32_b64_ddp2/rank_0/checkpoints/100000/pretrained_model 0 --pytorch_device cuda --headless True
# This command evaluates the DP policy trained for the `click_bell` task using the selected evaluation setting.
```

The evaluation results, including videos, will be saved in the `eval_result/{task_name}/dp/{setting}/{train_config_name}/{model_name}/{timestamp}/videos` directory under the project root. For DP, `train_config_name` is usually `None` unless you pass it explicitly through the evaluation config.
