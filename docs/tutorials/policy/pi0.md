# PI0
## Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies.
```bash
conda activate robosyn
# Install uv
pip install uv
```

Once uv is installed, run the following commands to set up the environment:
```bash
cd policy/pi0
apt install -y git-lfs build-essential pkg-config
GIT_LFS_SKIP_SMUDGE=1 uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## Generate RoboSynChallenge Data
See <a href="../collect_data.html">Collect Data Section</a> for more details.

## Prepare openpi0 Data for Training
`RoboSynChallenge` depends on the EmbodiChain emulator, which by default only supports acquiring Lerobot 3.0 data. We provide a script to convert Lerobot 3.0 data to Lerobot 2.1 data; this script is located in `scripts/convert_lerobot3.0_to_2.1.py`. Furthermore, our data collection script already supports one-click conversion; please refer to <a href="../collect_data.html">Collect Data Section</a> for details.

`convert_lerobot3.0_to_2.1.py` script will give you a processed dataset with the same name as the original {repo_id}, but in version 3.0, the dataset will be named {repo_id}_v3.0. Usage examples:
```python
python scripts/convert_lerobot3.0_to_2.1.py --repo-id {repo_id} --root /path/to/datasets
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

After preparing the data in lerobot2.1 format, create the `training_data` folders in the `policy/pi0` directory:

```
mkdir training_data
```
Then copy all the data you wish to use for training into `training_data/`.

## Write the Corresponding `train_config`
In `src/openpi/training/config.py`, there is a dictionary called `_CONFIGS`. You can modify a pre-configured PI0 configurations I’ve written:
`pi0_base_robosynchallenge_full`.

You only need to write `repo_id` on your datasets.(e.g., `repo_id=cobotmagic_Sim_beaker_mixer_dual`).

## Finetune model
```bash
# compute norm_stat for dataset
uv run scripts/compute_norm_stats.py --config-name ${train_config_name}
# uv run scripts/compute_norm_stats.py --config-name pi0_base_robosynchallenge_full

# train_config_name: The name corresponding to the config in _CONFIGS, such as pi0_base_robosynchallenge_full
# model_name: You can choose any name for your model
# gpu_use: if not using multi gpu,set to gpu_id like 0;else set like 0,1,2,3,4,5,6,7
bash finetune.sh ${train_config_name} ${model_name} ${gpu_use}
#bash finetune.sh pi0_base_aloha_robotwin_full click_bell 0,1,2,3,4,5,6,7
```

| Training mode | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Fine-Tuning (LoRA) | > 46 GB       | A6000(48G)           |
| Fine-Tuning (Full) | > 100 GB         | 2\*A100 (80GB) / 2\*H100 |

If your GPU memory is insufficient, please set the `fsdp_devices` parameter according to the following GPU memory reference, or reduce the `batch_size` parameter.
Or you can try setting `XLA_PYTHON_CLIENT_PREALLOCATE=false` in `finetune.sh`, it will cost lower gpu memory, but make training speed slower.

The default `batch_size` is 32 in the table below.

| GPU memory | Model type | GPU num |fsdp_devices | Example GPU |
| ----- | ----- | ----- | ----- | ----- |
|  40G | full | 4 | 4 | A100(40G)  |
|  80G | full | 2 | 2 | A100(80G)  |

## Eval on RoboSynChallenge

Checkpoints will be saved in policy/pi0/checkpoints/${train_config_name}/${model_name}/${checkpoint_id}

You can modify the `deploy_policy.yml` file to change the `checkpoint_id` you want to evaluate.
```
# ckpt_path like: policy/pi0/checkpoints/pi0_base_robosynchallenge_full/click_bell/30000
bash eval.sh ${task_name} [random | clear] ${train_config_name} ${model_name} ${gpu_id}
# bash eval.sh click_bell random pi0_base_robosynchallenge_full click_bell 0
# This command evaluate the policy($model_name) trained by the `click_bell_random` setting using the same `click_bell_random` setting.
```

The evaluation results, including videos, will be saved in the `eval_result/{task_name}/pi0/{setting}/{train_config_name}/{model_name}/{checkpoint_id}/` directory under the project root.
<!-- ### (1). Start OpenPI Server
Run this in one terminal. Replace `policy.dir` with the trained checkpoint step directory.

```bash
cd policy/pi0/
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config={train_config_name} \
    --policy.dir={policy_dir} \
    --port=8000
```

### (2). Run EmbodiChain Evaluation

Run this in a second terminal from the workspace root:

```bash
cd /path/to/RoboSynChallenge

python scripts/eval_openpi0_embodichain.py \
  --gym_config configs/{task_name}/gym_config.json \ # The gym_config file is exactly the same as during training.
  --action_config configs/{task_name}/action_config.json \ # The action_config file is exactly the same as during training.
  --num_envs 1 \
  --enable_rt \
  --host 127.0.0.1 \
  --port 8000 \
  --episodes 30 \ # Total number of episodes of evaluation
  --max_steps 500 \ # Maximum step per episode of evaluation
  --output results/{task_name}.json # Evaluation Results Statistics
  --filter_dataset_saving \ # Disabling data saving during evaluation
  --filter_visual_rand # Disabling visual domain randomization
#You can evaluate another EmbodiChain task by swapping `--gym_config` and `--action_config`, as long as the policy was trained for the same observation and action convention.
``` -->


