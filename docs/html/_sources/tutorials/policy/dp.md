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
uv run --locked --python 3.12 torchrun --standalone --nproc_per_node=2 \
  -m scripts.train_lerobot diffusion "${dataset}" "${output_dir}" \
  --batch_size=32 --policy.horizon=20 --policy.n_action_steps=8 \
  --steps=80000 --log_freq=100 --save_freq=10000
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
