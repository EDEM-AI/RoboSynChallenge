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

After data collection is completed, the collected data will be stored under `lerobot_dataset/{task_name}/`.

If you want to convert `lerobot 3.0` to the `lerobot 2.1` format manually, we have also provide ready-made conversion scripts:
```python
python scripts/convert_lerobot3.0_to_2.1.py --repo-id {repo_id} --root /path/to/datasets
```


If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), you can also use the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) to merge datasets.

Assume the two dataset directories are `/root/workspace/RoboSynChallenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Sim_beaker_mixer_dual` and `/root/workspace/RoboSynChallenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Real_beaker_mixer_dual`, you can use the following script and configuration file to merge it into `cobotmagic_merge_beaker_mixer_dual` in the same dir.