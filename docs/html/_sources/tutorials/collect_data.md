# Collect Data
We provide 1,000 pre-collected trajectories per task as part of the open-source release **RoboSynChallenge** Dataset. The datasets hosted on HuggingFace are available at [here](https://edem-ai.github.io/robosynchallenge.github.io/#/data).

However, we still strongly recommend users to perform data collection themselves.
```python
bash launch/run_task.sh {task_name} [random|clear] [3_0/2_1] [Other Extra Arguments]
# View supported tasks and extra arguments: bash launch/run_task.sh -h
# bash launch/run_task.sh click_bell clear 3_0
# Collect data for the click_bell task without domain randomization and the data is the LeRobot 3.0 format.
# bash launch/run_task.sh mixer_operating random 2_1
# Collect data for the mixer_operating task involving domain randomization and convert the data to the LeRobot 2.1 format.
```

The dataset task names are: `click_bell`, `handle_basket`, `water_pouring`, `table_rearrangement`, `items_handover`, `drawer_open_place`, `mixer_operating`, `item_assembly`, `manipulate_pipette`, `sample_loading`, and `open_pan`. For local collection, use `bash launch/run_task.sh -h` to view the tasks currently supported by the collection script.

After data collection is completed, the collected data will be stored under `lerobot_dataset/{task_name}/`.

If you want to convert `lerobot 3.0` to the `lerobot 2.1` format manually, we have also provide ready-made conversion scripts:
```python
python scripts/convert_lerobot3.0_to_2.1.py --repo-id {repo_id} --root /path/to/datasets
```


For pre-collected simulated and real datasets, see <a href="download_data.html">Download Data</a>.

If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), use the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) or the helper script [`launch/collect_combined_dataset.sh`](https://github.com/EDEM-AI/RoboSynChallenge/blob/main/launch/collect_combined_dataset.sh).
