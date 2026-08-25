# SmolVLA

## Environment Setup

First install RoboSynChallenge and EmbodiChain following the main installation
guide. For local installation, the recommended workspace layout is:

```bash
RoboSynChallenge_ws/
  EmbodiChain/
  RoboSynChallenge/
```

SmolVLA also needs LeRobot. The policy adapter supports either an installed
`lerobot` Python package or a local LeRobot source checkout.

During evaluation, there are two Python contexts:

- RoboSynChallenge/EmbodiChain Python runs the simulator and `scripts/eval_policy.py`.
- SmolVLA Python runs the LeRobot SmolVLA worker that loads the checkpoint.

They may be the same environment, but they do not have to be. If you use a
separate SmolVLA environment, set:

```bash
export SMOLVLA_PYTHON=$(which python)
export SMOLVLA_LEROBOT_ROOT=policy/smolvla/lerobot
```

If you only have the RoboSynChallenge repository, clone and install LeRobot from
the SmolVLA policy folder:

```bash
cd RoboSynChallenge
source ../EmbodiChain/.venv/bin/activate

bash policy/smolvla/setup_lerobot.sh
export SMOLVLA_LEROBOT_ROOT=$PWD/policy/smolvla/lerobot
export SMOLVLA_PYTHON=$(which python)
```

To use a specific LeRobot fork or revision:

```bash
LEROBOT_REPO=https://github.com/huggingface/lerobot.git \
LEROBOT_REF=main \
bash policy/smolvla/setup_lerobot.sh
```

## Generate RoboSynChallenge Data

See <a href="../collect_data.html">Collect Data Section</a> for details.
SmolVLA training assumes the task dataset is already exported in a valid
LeRobot format.

## Prepare SmolVLA Data for Training

SmolVLA training reads a RoboSynChallenge LeRobot dataset directly from
`--dataset.root`. You do not need to copy data into `policy/smolvla`.

For a single task, pass the task dataset directory directly:

```bash
datasets/cobotmagic_Sim_click_bell
```

The default feature mapping is:

```json
{
  "observation.qpos": "observation.state",
  "cam_high.color": "observation.images.camera1",
  "cam_left_wrist.color": "observation.images.camera2",
  "cam_right_wrist.color": "observation.images.camera3"
}
```

The policy consumes three RGB camera views and the robot joint state:

```text
cam_high.color
cam_left_wrist.color
cam_right_wrist.color
observation.qpos
```

If your dataset already uses SmolVLA feature names, override the rename map:

```bash
SMOLVLA_RENAME_MAP='{}' \
bash policy/smolvla/finetune.sh ${task_name} ${dataset_root} ${output_dir} ${gpu_id}
```

## Write the Corresponding Training Configuration

SmolVLA uses LeRobot's `lerobot-train` entrypoint. Most options are passed as
command-line overrides, so a new task usually does not require editing source
code.

Important defaults in `policy/smolvla/finetune.sh`:

| Flag | Default | Description |
| --- | --- | --- |
| `--policy.type` | `smolvla` | Selects the LeRobot SmolVLA policy. |
| `--policy.load_vlm_weights` | `true` | Loads the base VLM weights before finetuning. |
| `--policy.push_to_hub` | `false` | Disables automatic model upload during training. |
| `--wandb.enable` | `false` | Disables Weights & Biases by default. |
| `--policy.device` | `cuda` | Torch device used for training. |
| `--batch_size` | `32` | Per-process batch size. |
| `--steps` | `50000` | Total optimization steps. |
| `--num_workers` | `8` | DataLoader worker count. |
| `--dataset.repo_id` | `RoboSynChallenge/cobotmagic_Sim_<task_name>` | LeRobot repo id used by dataset metadata. |
| `--dataset.root` | Required | Local dataset root passed to `finetune.sh`. |
| `--output_dir` | Required | Directory where checkpoints and logs are saved. |
| `--job_name` | Output directory basename | Local training job name. |

Extra arguments after `gpu_id` are appended to `lerobot-train`, so they can
override these defaults.

## Finetune Model

Make sure the shell can find `lerobot-train`. This is true if LeRobot is
installed in the active environment, or if `SMOLVLA_LEROBOT_ROOT` points to a
local LeRobot source checkout.

```bash
cd RoboSynChallenge

task_name=click_bell
dataset_root=datasets/cobotmagic_Sim_${task_name}
output_dir=outputs/train/cobotmagic_smolvla_${task_name}_run1
gpu_id=0

bash policy/smolvla/finetune.sh ${task_name} ${dataset_root} ${output_dir} ${gpu_id} \
  --steps=50000 \
  --batch_size=32 \
  --num_workers=8 \
  --persistent_workers=true
```

Run finetuning in the background:

```bash
SMOLVLA_NOHUP=1 bash policy/smolvla/finetune.sh \
  ${task_name} ${dataset_root} ${output_dir} ${gpu_id} \
  --steps=50000

tail -f outputs/train/logs/$(basename ${output_dir}).log
```

Override the dataset repo id if your local metadata uses a custom name:

```bash
SMOLVLA_DATASET_REPO_ID=my_org/my_dataset \
bash policy/smolvla/finetune.sh ${task_name} ${dataset_root} ${output_dir} ${gpu_id}
```

Checkpoints are saved under:

```text
${output_dir}/checkpoints/<step>/pretrained_model
```

## Eval on RoboSynChallenge

Use one of these task names:

```text
click_bell
handle_basket
water_pouring
table_rearrangement
items_handover
drawer_open_place
mixer_operating
item_assembly
manipulate_pipette
sample_loading
```

Download a released SmolVLA checkpoint from Hugging Face:

```bash
cd RoboSynChallenge
mkdir -p checkpoints

task_name=click_bell
hf download RoboSynChallenge/SmolVLA_sim_${task_name} \
  --repo-type model \
  --local-dir checkpoints/SmolVLA_sim_${task_name}
```

The released repository contains the complete LeRobot `pretrained_model`
contents, so pass the downloaded directory directly as `checkpoint_path`:

```bash
checkpoint_path=checkpoints/SmolVLA_sim_${task_name}

bash policy/smolvla/eval.sh ${task_name} random ${checkpoint_path} ${gpu_id} \
  --pytorch_device cuda \
  --headless true \
  --renderer auto \
  --max_episodes 20 \
  --eval_video_log true \
  --smolvla_rescale_gripper true
```

Example:

```bash
bash policy/smolvla/eval.sh click_bell random checkpoints/SmolVLA_sim_click_bell 0 \
  --pytorch_device cuda \
  --headless true \
  --renderer auto \
  --max_episodes 20 \
  --eval_video_log true \
  --smolvla_steps 10 \
  --smolvla_rescale_gripper true
```

The evaluation script also accepts a LeRobot checkpoint step directory:

```bash
bash policy/smolvla/eval.sh click_bell random \
  outputs/train/cobotmagic_smolvla_click_bell_run1/checkpoints/050000 \
  0 \
  --pytorch_device cuda \
  --headless true
```

Evaluation results, including videos, are saved under:

```text
eval_result/{task_name}/smolvla/{setting}/{train_config_name}/{model_name}/{timestamp}/videos
```

Runtime notes:

- `gpu_id` is the physical GPU id used by DexSim and EmbodiChain.
- Do not manually set `CUDA_VISIBLE_DEVICES` before evaluation. The script keeps
  the simulator parent process unmasked and masks only the SmolVLA worker.
- `--smolvla_steps` controls how many low-level actions are executed per policy
  query. The default is `10`.
- `--smolvla_rescale_gripper true` rescales normalized gripper outputs from
  `[0, 1]` into the environment action range. Use `auto` to infer this from
  checkpoint stats.
- The effective episode horizon follows the task `gym_config.max_episode_steps`
  in RoboSynChallenge. Depending on the repository version, command-line
  `--max_steps` may be used only as a fallback when the gym config does not set
  a timeout.

## Troubleshooting

If evaluation cannot import `lerobot`, install LeRobot in the worker Python
environment or set:

```bash
export SMOLVLA_PYTHON=$(which python)
export SMOLVLA_LEROBOT_ROOT=policy/smolvla/lerobot
```

If `lerobot-train` is not on `PATH`, run:

```bash
bash policy/smolvla/setup_lerobot.sh
export SMOLVLA_LEROBOT_ROOT=$PWD/policy/smolvla/lerobot
```

If Hugging Face downloads are slow or rate-limited:

```bash
export HF_HOME=.cache/huggingface
hf auth login
```

If DexSim reports an invalid CUDA device ordinal, remove manual
`CUDA_VISIBLE_DEVICES` settings and pass the physical GPU id as the fourth
argument to `policy/smolvla/eval.sh`.
