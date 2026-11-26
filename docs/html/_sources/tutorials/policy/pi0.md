# PI0

The original OpenPI/JAX trainer and Orbax checkpoint format run in the root Python 3.12 environment. LeRobot 0.6.1 reads v3.0 dataset frames, camera streams, and task prompts. The OpenPI observation and action transforms, normalization assets, and model parameter layout still apply to existing runs.

The RoboSynChallenge training configuration uses
`han-xudong/cobotmagic_Sim_click_bell` at `v3.0` by default. The OpenPI
base weights use `OPENPI_DATA_HOME`. See [data and checkpoints](../download_data.md).

```bash
uv sync --locked --python 3.12
export OPENPI_DATA_HOME=/data/openpi-cache
bash launch/openpi.sh pi0 stats --config-name pi0_base_robosynchallenge_full \
  --asset-id cobotmagic
bash policy/pi0/finetune.sh pi0_base_robosynchallenge_full my-run 3 \
  --data.assets.asset-id cobotmagic --batch-size 1
```

Select another Hub dataset with `--repo-id` for stats and `--data.repo-id`
for training. For a local dataset, point `--dataset-root` or
`--data.dataset-root` at the directory containing `meta/info.json`.
Compute normalization stats with the same dataset and asset ID used in training.

## Resume an existing run

Pass the original experiment name, config, normalization asset ID and checkpoint base directory. Increase `--num-train-steps` to the desired total step count.

```bash
bash policy/pi0/finetune.sh pi0_base_robosynchallenge_full my-run 3 \
  --resume --num-train-steps 40000 --data.dataset-root /data/task-dataset \
  --data.repo-id local/robosynchallenge --data.assets.asset-id cobotmagic --batch-size 1
```

## Evaluate a saved Orbax checkpoint

```bash
bash policy/pi0/eval.sh click_bell clear pi0_base_robosynchallenge_full my-run 3 \
  --checkpoint_id 20000 --headless true --renderer fast-rt
```

The default lookup is `policy/pi0/checkpoints/<config>/<experiment>/<step>`. Use `--checkpoint_path` for a checkpoint stored elsewhere and `--asset_id` if the checkpoint contains several normalization assets.
