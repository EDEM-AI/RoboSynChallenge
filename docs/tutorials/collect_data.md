# Collect Data
Released LeRobot v3.0 datasets are listed in the
[Real data](https://huggingface.co/collections/han-xudong/real-data-6b078018b5eb646b7c26fbe1)
and [Sim data](https://huggingface.co/collections/han-xudong/sim-data-6b07801c45756c22865cc334)
collections. See [download instructions](download_data.md) for the pinned revision.

The native collection pipeline writes LeRobot v3.0. Each official task has a CobotMagic deployment for both `clear` and `random`.

```bash
uv sync --locked --python 3.12
bash launch/run_task.sh click_bell clear --headless --max_episodes 1
bash launch/collect_combined_dataset.sh click_bell 10 10 /data/click_bell-combined --headless
uv run --locked --python 3.12 embodichain dataset validate /data/click_bell-combined
```

The task recorder defaults to `lerobot_dataset/<task_name>/`. Pass `--dataset-dir` to select another location. Use the importer to read a local legacy release and create a v3.0 copy.

```bash
uv run --locked --python 3.12 python -m scripts.import_lerobot_v21 /data/legacy /data/imported-v30
```

## Collect simulation data with depth information

Set `enable_depth: true` on the camera in the robot component referenced by
`task.cobotmagic.yaml`. For `click_bell`, both settings use
`configs/click_bell/cobotmagic.yaml`. For example, add the flag to the existing
`cam_high` entry under `sensor`.

```yaml
sensor:
  - sensor_type: Camera
    uid: cam_high
    enable_depth: true
```

The native recorder saves depth as a numeric feature by default. Keep the RGB
camera fields enabled if the dataset will also be used to train the policy.

For pre-collected simulated and real datasets, see <a href="download_data.html">Download Data</a>.

If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), use the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) or the helper script [`launch/collect_combined_dataset.sh`](https://github.com/EDEM-AI/RoboSynChallenge/blob/main/launch/collect_combined_dataset.sh).
