# Collect Data
We provide 1,000 pre-collected trajectories per simulation task as part of the open-source release **RoboSynChallenge** Dataset. The datasets hosted on Hugging Face are available in the [Sim Data](https://huggingface.co/collections/han-xudong/sim-data-6b07801c45756c22865cc334) and [Real Data](https://huggingface.co/collections/han-xudong/real-data-6b078018b5eb646b7c26fbe1) collections.

However, we still strongly recommend users to perform data collection themselves. Both the `clear` and `random` settings write LeRobot v3.0.

```bash
# View supported tasks and extra arguments.
bash launch/run_task.sh -h
# Collect data for click_bell without domain randomization.
bash launch/run_task.sh click_bell clear --headless --max_episodes 1
# Collect data for mixer_operating with domain randomization.
bash launch/run_task.sh mixer_operating random --headless --max_episodes 1
```

After data collection is completed, the collected data will be stored under `lerobot_dataset/<task_name>/`. Pass `--dataset-dir` to select another location. Use the importer to read a local legacy release and create a v3.0 copy.

```bash
uv run --locked --python 3.12 python -m scripts.import_lerobot_v21 datasets/legacy datasets/imported-v30
```

## 🔧 Data Collection Hardware

For GPUs equivalent to the RTX 5060 Ti, collect large datasets in batches. The original collection guide suggests starting with at most about 500 episodes per batch on this tier. Run the merge script to combine data from separate collection batches. Validate the resulting dataset before training.

```bash
bash launch/collect_combined_dataset.sh click_bell 10 10 datasets/click_bell-combined --headless
uv run --locked --python 3.12 embodichain dataset validate datasets/click_bell-combined
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

For an explicitly configured recorder, include the depth observation in its feature mapping. Keep the RGB camera fields enabled if the dataset will also be used to train the policy.

For pre-collected simulated and real datasets, see <a href="download_data.html">Download Data</a>.

If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), use the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) or the helper script [`launch/collect_combined_dataset.sh`](https://github.com/EDEM-AI/RoboSynChallenge/blob/main/launch/collect_combined_dataset.sh).
