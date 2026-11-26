# ACT

Install the repository's root environment with `uv sync --locked --python 3.12`. A task name selects the published simulation dataset at `v3.0`.

```bash
CUDA_VISIBLE_DEVICES=3 bash launch/train_policy.sh act click_bell /data/act-run \
  --steps=10000 --save_freq=1000 --wandb.enable=false
bash launch/eval_policy.sh act click_bell clear 3 \
  --headless true --renderer fast-rt --max_episodes 10
```

The dataset argument also accepts a full Hub ID or the local directory
containing `meta/info.json`, `data/` and `videos/`. Extra training flags
are passed to `lerobot-train`, including `--policy.path` for compatible
base weights and `--batch_size` for memory control.

The evaluation example loads `han-xudong/ACT_sim_click_bell`. Pass a
checkpoint directory or Hub ID before the GPU argument to select another
checkpoint. See [data and checkpoint downloads](../download_data.md).
