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

If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), merge them with the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) or [`launch/collect_combined_dataset.sh`](https://github.com/EDEM-AI/RoboSynChallenge/blob/main/launch/collect_combined_dataset.sh) before placing the result in this policy's training data folder.

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