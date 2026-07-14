# ACT
## Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies.
```bash
conda activate robosyn
# Install uv
pip install uv
```

Once uv is installed, run the following commands to set up the environment:
```bash
cd policy/act
uv sync --frozen
source .venv/bin/activate
```

The ACT environment is defined by `policy/act/pyproject.toml` and `policy/act/uv.lock`. If you are running on a machine without network access during training or evaluation, activate the environment first and use `uv run --locked --no-sync --python .venv/bin/python ...`.

## Generate RoboSynChallenge Data
See <a href="../collect_data.html">Collect Data Section</a> for more details.

## Prepare ACT Data for Training
ACT training reads a RoboSynChallenge LeRobot dataset directly from `--dataset-root`. The wrapper in `policy/act/scripts/train.py` maps the RoboSynChallenge dataset feature names to the LeRobot ACT feature names expected by the installed LeRobot policy.

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

After preparing the data, keep the dataset directory available and pass it to `finetune.sh`. You do not need to copy the data into `policy/act`.

## Write the Corresponding `train_config`
ACT uses the LeRobot `ACTConfig` assembled by `policy/act/scripts/train.py`. You usually do not need to edit source code for a new task. Instead, pass the dataset path, output path, and training arguments from the command line.

Training arguments are configured from the command line:

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset-root` | Required | Path to the RoboSynChallenge LeRobot dataset directory. |
| `--repo-id` | Dataset directory name | LeRobot repo id used for metadata; normally leave unset for local training. |
| `--output-dir` | Required | Directory where checkpoints, logs, and config snapshots are saved. |
| `--job-name` | `None` | Optional local job name when `--wandb-name` is not set. |
| `--video-backend` | `pyav` | Video decoding backend passed to LeRobot. |
| `--device` | `cuda` | Torch device used by the ACT policy. |
| `--batch-size` | `8` | Per-process batch size. In DDP, global batch size is `batch-size * number_of_processes`. |
| `--num-workers` | `4` | DataLoader worker count per process. |
| `--steps` | `100000` | Total optimization steps. |
| `--log-freq` | `200` | Console and tracker logging interval in steps. |
| `--save-freq` | `20000` | Checkpoint save interval in steps. |
| `--eval-freq` | `0` | LeRobot training-time eval interval. `0` disables it. |
| `--seed` | `1000` | Random seed. |
| `--n-obs-steps` | `1` | Number of observation steps consumed by ACT. |
| `--chunk-size` | `16` | Number of actions predicted per policy chunk. |
| `--n-action-steps` | `8` | Number of predicted actions executed before the next policy query. |
| `--use-amp` | Off | Enable automatic mixed precision. |
| `--wandb` | Off | Enable Weights & Biases logging. |
| `--wandb-project` | `robosynchallenge` | Weights & Biases project name. |
| `--wandb-name` | `None` | Weights & Biases run name; also used as `job_name` if set. |
| `--resume` | Off | Resume from the existing output directory. |
| `--overwrite` | Off | Delete the output directory before training when not resuming. |
| `--no-imagenet-stats` | Off | Disable ImageNet normalization stats in the LeRobot dataset config. |
| `--no-save-checkpoint` | Off | Disable checkpoint writing. |
| `--distributed` | Off | Enable DDP wrapping. Use with `torchrun`. |
| `--local-rank`, `--local_rank` | From `LOCAL_RANK` | Local GPU rank supplied by `torchrun`; usually do not set manually. |

For multi-GPU training, launch `scripts/train.py` with `torchrun`, pass `--distributed`, and set the per-process `--batch-size`. For example, global batch size 64 on 2 GPUs uses `--batch-size 32`.

## Finetune model
```bash
# dataset_root: path to the RoboSynChallenge LeRobot dataset
# output_dir: where checkpoints will be saved
# gpu_use: if not using multi gpu, set to gpu_id like 0; else set like 0,1
bash policy/act/finetune.sh ${dataset_root} ${output_dir} ${gpu_use} \
  --steps 80000 \
  --batch-size 64 \
  --chunk-size 50 \
  --n-action-steps 50 \
  --log-freq 100 \
  --save-freq 10000 \
  --wandb \
  --overwrite
```

For 2-GPU training with global batch size 64:
```bash
cd policy/act
# Select the physical GPUs visible to this training job.
export CUDA_VISIBLE_DEVICES=0,1
# These NCCL settings are useful on workstations or containers where peer-to-peer
# or shared-memory transport is unstable. Remove them if your cluster requires
# the default NCCL transport.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
# Surface distributed failures promptly instead of hanging silently.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# torchrun sets RANK, WORLD_SIZE, and LOCAL_RANK for scripts/train.py.
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --distributed \
  --dataset-root ${dataset_root} \
  --output-dir ${output_dir} \
  --batch-size 32 \
  --chunk-size 50 \
  --n-action-steps 50 \
  --steps 80000 \
  --log-freq 100 \
  --save-freq 10000 \
  --wandb \
  --wandb-project robosynchallenge \
  --wandb-name ${run_name} \
  --overwrite
```

| Training mode | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Single GPU ACT | > 20 GB | RTX 4090 / RTX 5090 |
| 2-GPU ACT | > 20 GB per GPU | 2\*RTX 4090 / 2\*RTX 5090 |

If your GPU memory is insufficient, reduce `--batch-size`, reduce `--chunk-size`, or enable multi-GPU training with a smaller per-GPU batch size.

The default `batch_size` in `scripts/train.py` is 8, but the recommended RoboSynChallenge ACT training command uses batch size 64.

| Global batch size | GPU num | Per-GPU `--batch-size` | Example GPU |
| ----- | ----- | ----- | ----- |
| 64 | 1 | 64 | RTX 5090 |
| 64 | 2 | 32 | 2\*RTX 5090 |
| 64 | 4 | 16 | 4\*RTX 5090 |

## Eval on RoboSynChallenge

Checkpoints will be saved in `${output_dir}/checkpoints/${checkpoint_id}/pretrained_model` for single-GPU training. In distributed training, every rank writes a full checkpoint under `${output_dir}/rank_${rank}/checkpoints/${checkpoint_id}/pretrained_model`; use `rank_0` for evaluation and release.

```
# checkpoint_path like: checkpoints/act_click_bell_c50_a50_b64_ddp2/rank_0/checkpoints/080000/pretrained_model
bash policy/act/eval.sh ${task_name} [random | clear | random_eval_once] ${checkpoint_path} ${gpu_id} \
  --pytorch_device cuda \
  --headless True
# bash policy/act/eval.sh click_bell random_eval_once checkpoints/act_click_bell_c50_a50_b64_ddp2/rank_0/checkpoints/080000/pretrained_model 0 --pytorch_device cuda --headless True
# This command evaluates the ACT policy trained for the `click_bell` task using the selected evaluation setting.
```

The evaluation results, including videos, will be saved in the `eval_result/{task_name}/act/{setting}/{train_config_name}/{model_name}/{timestamp}/videos` directory under the project root. For ACT, `train_config_name` is usually `None` unless you pass it explicitly through the evaluation config.
