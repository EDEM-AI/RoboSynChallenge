# Motus
## Environment Setup
[TODO]

## Generate RoboSynChallenge Data
See <a href="../collect_data.html">Collect Data Section</a> for more details.

## Prepare Motus Data for Training
[TODO]

`RoboSynChallenge` depends on the EmbodiChain emulator, which by default only supports acquiring Lerobot 3.0 data. We provide a script to convert Lerobot data to Motus data; this script is located in `xxx`.

Usage examples:
```python

```

If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), you can also use the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) to merge datasets. Here, we provide an example of using lerobot-edit-datase to merge datasets:

Assume the two dataset directories are `/root/workspace/RoboSynChallenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Sim_beaker_mixer_dual` and `/root/workspace/RoboSynChallenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Real_beaker_mixer_dual`, you can use the following script and configuration file to merge it into `cobotmagic_merge_beaker_mixer_dual` in the same dir.
First, you can create a merge_config.json
```
{
  "repo_id": "lerobot_dataset/cobotmagic_merge_beaker_mixer_dual",
  "push_to_hub": false,
  "operation": {
    "type": "merge",
    "repo_ids": [
      "lerobot_dataset/cobotmagic_Sim_beaker_mixer_dual",
      "lerobot_dataset/cobotmagic_Real_beaker_mixer_dual"
    ]
  }
}
```
Then, use the following code:
```shell
export HF_LEROBOT_HOME=/root/workspace/RoboSynChallenge/
lerobot-edit-dataset --config_path /root/workspace/RoboSynChallenge/merge_config.json
```

After preparing the data in motus format, create the `training_data` folders in the `policy/motus` directory:

```
mkdir training_data
```
Then copy all the data you wish to use for training into `training_data/`.

## Finetune model
[TODO]
```bash

```

## Eval on RoboSynChallenge
[TODO]
The evaluation results, including videos, will be saved in the `eval_result/{task_name}/motus/{setting}/{train_config_name}/{model_name}/{checkpoint_id}/` directory under the project root.