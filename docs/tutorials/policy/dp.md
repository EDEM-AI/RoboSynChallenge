# DP

## Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. Once uv is installed, run the following command from the RoboSynChallenge root:

```bash
uv sync --locked --python 3.12
```

See [Installation](../../getting_started/installation.md) for Docker and local uv setup.

## Generate RoboSynChallenge Data

See <a href="../collect_data.html">Collect Data Section</a> for more details.

## Prepare DP Data for Training

DP training reads a RoboSynChallenge LeRobot v3.0 dataset directly. For a single task, pass the task dataset directory directly. A task name such as `click_bell` selects the published simulation dataset at `v3.0`. You can also pass a full Hub ID.

```text
lerobot_dataset/han-xudong/cobotmagic_Sim_click_bell
```

If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), you can use the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) to merge datasets. After preparing the data, keep the dataset directory available and pass it to the training launcher.

## Write the Corresponding `train_config`

You usually do not need to edit source code for a new task. Instead, pass the dataset path, output path, and training arguments from the command line. The root launcher forwards native options to `lerobot-train`. Use `--policy.path` to initialize from compatible base weights.

The Python entry point `policy.dp.scripts.train` also accepts the original command-line options:

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset-root` | Required | Path to the RoboSynChallenge LeRobot dataset directory. |
| `--repo-id` | Inferred from local metadata | LeRobot repo id used for metadata; normally leave unset for local training. |
| `--output-dir` | Required | Directory where checkpoints, logs, and config snapshots are saved. |
| `--job-name` | `None` | Optional local job name when `--wandb-name` is not set. |
| `--video-backend` | `pyav` | Video decoding backend passed to LeRobot. |
| `--device` | `cuda` | Torch device used by the DP policy. |
| `--batch-size` | `8` | Per-process batch size. In DDP, global batch size is `batch-size * number_of_processes`. |
| `--num-workers` | `4` | DataLoader worker count per process. |
| `--steps` | `100000` | Total optimization steps. |
| `--log-freq` | `200` | Console and tracker logging interval in steps. |
| `--save-freq` | `20000` | Checkpoint save interval in steps. |
| `--eval-freq` | `0` | LeRobot training-time eval interval. `0` disables it. |
| `--seed` | `1000` | Random seed. |
| `--n-obs-steps` | `2` | Number of observation steps consumed by DP. |
| `--horizon` | `16` | Diffusion prediction horizon. |
| `--n-action-steps` | `8` | Number of predicted actions executed before the next policy query. |
| `--num-inference-steps` | `None` | Number of denoising steps at inference time. Leave unset to use the LeRobot default. |
| `--crop-shape HEIGHT WIDTH` | `None` | Crop RGB observations before the DP RGB encoder; omit to use full images. |
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

## Finetune model

```bash
# dataset: a task name, a Hub repo ID, or a local dataset directory
# output_dir: where checkpoints will be saved
CUDA_VISIBLE_DEVICES=0 bash launch/train_policy.sh diffusion "${dataset}" "${output_dir}" \
  --steps=80000 --batch_size=32 --policy.horizon=20 \
  --policy.n_action_steps=8 --log_freq=100 --save_freq=10000
```

For 2-GPU training with global batch size 64, start from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0,1
uv run --locked --python 3.12 torchrun --standalone --nproc_per_node=2 -m policy.dp.scripts.train \
  --distributed --dataset-root "${dataset_root}" --output-dir "${output_dir}" \
  --batch-size 32 --horizon 20 --n-action-steps 8 \
  --steps 80000 --log-freq 100 --save-freq 10000
```

If your GPU memory is insufficient, reduce the per-GPU batch size, reduce the policy's action window, or use multi-GPU training with a smaller per-GPU batch size.

## Eval on RoboSynChallenge

Checkpoints are saved in `${output_dir}/checkpoints/${checkpoint_id}/pretrained_model`. Find the tasks with released weights in the [model checkpoints collection](../download_data.md#checkpoints).

Download a released DP checkpoint into a task-specific local directory:

```bash
task_name=click_bell
hf download "han-xudong/DP_sim_${task_name}" --repo-type model \
  --local-dir "checkpoints/DP_sim_${task_name}"
```

The released repository contains the complete `pretrained_model` directory, so pass the downloaded directory directly as `${checkpoint_path}`:

```bash
checkpoint_path="checkpoints/DP_sim_${task_name}"
bash policy/dp/eval.sh "${task_name}" random "${checkpoint_path}" 0 \
  --pytorch_device cuda --headless true
```

To load the published checkpoint automatically, use `bash launch/eval_policy.sh dp click_bell clear 0 --headless true`.

The evaluation results, including videos, are saved under `eval_result/{task_name}/dp/{setting}/{train_config_name}/{model_name}/{timestamp}/`.
