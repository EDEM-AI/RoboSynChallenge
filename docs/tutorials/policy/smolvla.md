# SmolVLA

## Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. Once uv is installed, run the following command from the RoboSynChallenge root:

```bash
uv sync --locked --python 3.12
```

See [Installation](../../getting_started/installation.md) for Docker and local uv setup.

## Generate RoboSynChallenge Data

See <a href="../collect_data.html">Collect Data Section</a> for more details.

## Prepare SmolVLA Data for Training

SmolVLA training reads a RoboSynChallenge LeRobot v3.0 dataset directly. The dataset argument accepts a task name, a full Hub ID, or the local dataset directory. The policy consumes three RGB camera views and the robot joint state.

## Write the Corresponding Training Configuration

SmolVLA uses LeRobot's `lerobot-train` entry point. Most options are passed as command-line overrides, so a new task usually does not require editing source code.

## Finetune Model

```bash
CUDA_VISIBLE_DEVICES=0 bash launch/train_policy.sh smolvla click_bell outputs/smolvla-run \
  --steps=10000 --save_freq=1000 --wandb.enable=false
```

When initializing from the published task checkpoint, map the dataset's camera names to the checkpoint's input features.

```bash
CUDA_VISIBLE_DEVICES=0 bash launch/train_policy.sh smolvla click_bell outputs/smolvla-finetune \
  --policy.path=han-xudong/SmolVLA_sim_click_bell \
  --rename_map='{"observation.images.cam_high":"observation.images.camera1","observation.images.cam_left_wrist":"observation.images.camera2","observation.images.cam_right_wrist":"observation.images.camera3"}' \
  --batch_size=1
```

Checkpoints are saved under `${output_dir}/checkpoints/<step>/pretrained_model`.

## Eval on RoboSynChallenge

```bash
bash launch/eval_policy.sh smolvla click_bell clear 0 \
  --headless true --renderer fast-rt --max_episodes 10
```

The evaluation example loads `han-xudong/SmolVLA_sim_click_bell`. Pass a
checkpoint directory or Hub ID before the GPU argument to select another
checkpoint. See [data and checkpoint downloads](../download_data.md).

SmolVLA's interpreter, worker and simulator use the same root environment and visible GPUs. The first run downloads its model assets unless they are already cached. A checkpoint includes the saved preprocessor and postprocessor.

## Troubleshooting

If model downloads are slow or rate-limited, configure `HF_HOME` and run `hf auth login` for gated assets. Pass the physical GPU ID to the evaluation launcher as shown above.
