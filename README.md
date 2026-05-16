<div align="center">
<h1>RoboSynChallenge</h1>

[[Website]](https://edem-ai.github.io/robosynchallenge.github.io/)
______________________________________________________________________
![image](misc/robosynchallenge-pipeline.png)

</div>

---

# Contents

- [Installation](#Installation)
- [Datasets](#Dataset)
- [Getting Started](#Getting-Started)
  - [Task](#Implement-a-Task)
  - [Training](#Training)
  - [Evaluation](#Evaluation)

# Installtion
## Docker (Recommended)
Please run the following commands in the given order to install the dependency for **EmbodiChain**.
```
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8

mkdir RoboSynChallenge_ws && cd RoboSynChallenge_ws
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
./docker/docker_run.sh <container_name> <data_path>

conda activate py310
cd /path/to/EmbodiChain
pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
```

Then install the `RoboSynChallenge` package:
```
cd /path/to/RoboSynChallenge
pip install -e .
```

# Datasets
We provide 1,000 pre-collected trajectories per task as part of the open-source release **RoboSynChallenge** Dataset. The datasets hosted on HuggingFace are available at [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).

However, we still strongly recommend users to perform data collection themselves.
```python
bash launch/[random|clear]/**.sh
# Clean Data Example: bash launch/clear/beaker_mixer_duel.sh
# Radomized Data Example: bash launch/random/beaker_mixer_duel.sh
```
After data collection is completed, the collected data will be stored under `lerobot_dataset/{task_name}/`.

An episode's data will be stored in the `lerobot 3.0` format. If you want to convert to the `lerobot 2.1` format, we have also provide ready-made conversion scripts:
```python
python scripts/convert_lerobot3.0_to_2.1.py --repo-id {repo_id} --root /path/to/datasets
```

# Getting Started
In the following, we provide example scripts for implement a task, training and evaluation.
## Implement a Task

1. Create a new task environment class in `robosynchallenge/tasks/{task_name}.py` that inherits from `EmbodiedEnv`.
2. Create a configuration file in `configs/{task_name}/xxx.json` that defines the environment and robot setup.
3. Implement the `create_demo_action_list()` method in your task environment to generate demonstration actions based on the task requirements.

References:
- [EmbodiChain Documentation](https://dexforce.github.io/EmbodiChain/overview/gym/env.html)


## Training
We will use OpenPI as an example to explain how to use data for training and explain three steps:
1. Prepare dataset
2. Define training configs and running training

### 1. Prepare dataset
The default configuration provided by RoboSynChallenge supports data collection from Lerobot 3.0, and we provide a script for converting LeRobot 3.0 data to LeRobot 2.1 in [`scripts/convert_lerobot3.0_to_2.1.py`](scripts/convert_lerobot3.0_to_2.1.py).
If you want to train on multiple datasets together (e.g., multi-task, mixed training with simulated and real data), you can also use the [lerobot-edit-dataset tool](https://huggingface.co/docs/lerobot/using_dataset_tools) to merge datasets.
Assume the two dataset directories are `/root/workspace/Embodied_Challenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Sim_beaker_mixer_dual` and `/root/workspace/Embodied_Challenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Real_beaker_mixer_dual`, you can use the following script and configuration file to merge it into `cobotmagic_merge_beaker_mixer_dual` in the same dir.
Create a merge_config.json
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

```shell
export HF_LEROBOT_HOME=/root/workspace/Embodied_Challenge/
lerobot-edit-dataset --config_path /root/workspace/Embodied_Challenge/merge_config.json
```

### 2. Define training configs and running training

To fine-tune a base model, you need to define configs for data processing and training. We provide example fine-tuning configs for π₀ on `RoboSynChallenge` dataset. For more detailed configuration instructions, please refer to [openpi](https://github.com/Physical-Intelligence/openpi).

#### Data Config Example
If you are using the default configuration environment provided by `RoboSynChallenge` to collect data, you can copy this class to [policies/libero_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py)
```python
@dataclasses.dataclass(frozen=True)
class EmbodiChainInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:

        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class EmbodiChainOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
```
If you are using the default configuration environment provided by `RoboSynChallenge` to collect data, you can copy this class to [training/config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
```python
@dataclasses.dataclass(frozen=True)
class LeRobotEmbodiChainDataConfig(DataConfigFactory):
    """
    If you are using the default configuration environment provided by RoboSynChallenge to collect data, you can copy this class.
    """

    extra_delta_transform: bool = False
    action_sequence_keys: Sequence[str] = ("action",)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "cam_high.color",
                        "observation/left_wrist_image": "cam_left_wrist.color",
                        "observation/right_wrist_image": "cam_right_wrist.color",
                        "observation/state": "observation.qpos",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.EmbodiChainInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.EmbodiChainOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
```
#### Train Config Example
We provide a training configuration example.
```python
TrainConfig(
        name="pi0_robosyncchallenge_beaker_mixer_dual",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=LeRobotEmbodiChainDataConfig(
            repo_id="random/cobotmagic_Sim_beaker_mixer_dual",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        batch_size=32,
        pytorch_weight_path="/root/.cache/openpi/openpi-assets/checkpoints/pi0_base_torch",
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        wandb_enabled=True,
    ),
```

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_robosyncchallenge_beaker_mixer_dual
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_robosyncchallenge_beaker_mixer_dual --exp-name=my_experiment --overwrite
```

## Evaluation
We offer an evaluation script for you to evaluate pi0 models separately. If you want to eval your own model, you can refer to this [eval_script](https://github.com/wuxinxin27/RoboSynChallenge/blob/main/scripts/eval_openpi0_embodichain.py) for implementation.

```shell
python scripts/eval_openpi0_embodichain.py --benchmark BENCHMARK_NAME \
                                   --task_id TASK_ID \
                                   --algo ALGO_NAME \
                                   --policy POLICY_NAME \
                                   --seed SEED \
                                   --ep EPOCH \
                                   --load_task LOAD_TASK \
                                   --device_id CUDA_ID
```
### (1). Start OpenPI Server
Run this in one terminal. Replace `policy.dir` with the trained checkpoint step directory.

```bash
cd /path/tp/openpi
uv run openpi/scripts/serve_policy.py policy:checkpoint --policy.config={TRAIN_CONFIG_NAME} --policy.dir={POLICY_DIR} --port=8000
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
```

## LeaderBoard
The full leaderboard and setting can be found in: https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard.