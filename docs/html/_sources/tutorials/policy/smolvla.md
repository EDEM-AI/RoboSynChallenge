# SmolVLA

Install the repository's root environment with `uv sync --locked --python 3.12`. A task name selects the published simulation dataset at `v3.0`.

```bash
CUDA_VISIBLE_DEVICES=3 bash launch/train_policy.sh smolvla click_bell /data/smolvla-run \
  --steps=10000 --save_freq=1000 --wandb.enable=false
bash launch/eval_policy.sh smolvla click_bell clear 3 \
  --headless true --renderer fast-rt --max_episodes 10
```

The dataset argument also accepts a full Hub ID or a local v3.0 directory.
When initializing from the published task checkpoint, map the dataset's
camera names to the checkpoint's input features.

```bash
CUDA_VISIBLE_DEVICES=3 bash launch/train_policy.sh smolvla click_bell /data/smolvla-finetune \
  --policy.path=han-xudong/SmolVLA_sim_click_bell \
  --rename_map='{"observation.images.cam_high":"observation.images.camera1","observation.images.cam_left_wrist":"observation.images.camera2","observation.images.cam_right_wrist":"observation.images.camera3"}' \
  --batch_size=1
```

The evaluation example loads `han-xudong/SmolVLA_sim_click_bell`. Pass a
checkpoint directory or Hub ID before the GPU argument to select another
checkpoint. See [data and checkpoint downloads](../download_data.md).

SmolVLA's interpreter, worker and simulator use the same root environment and visible GPUs. The first run downloads its model assets unless they are already cached. A checkpoint includes the saved preprocessor and postprocessor.
