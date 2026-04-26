# Embodied_Challenge/scripts 目录说明

本 README 统计了本目录下每个脚本的用途、使用方法及详细使用案例，便于快速查阅和使用。

---

## 1. camera_extrinsics_to_lootat.py

**用途**：
将相机的变换矩阵（外参）转换为 EmbodiChain 所需的 camera extrinsics（eye/target/up 格式），并输出可直接粘贴到配置文件的片段。

**使用方法**：
1. 编辑脚本开头 USER SETTINGS 区域，填写你的相机与机械臂的变换矩阵。
2. 直接运行脚本：
   ```bash
   python camera_extrinsics_to_lootat.py
   ```
3. 脚本会输出 eye/target/up 以及可粘贴的 extrinsics 配置片段。

**详细案例**：
- 修改 `T_ARM_CAM` 和 `T_WORLD_ARM` 为你的实际标定外参。
- 运行后输出如下：
  ```json
  # Computed camera extrinsics
  {
    "eye": [...],
    "target": [...],
    "up": [...]
  }
  # Ready-to-paste config snippet
  {
    "extrinsics": {"eye": [...], "target": [...], "up": [...]}
  }
  # Derived T_world_cam
  [...矩阵...]
  ```

---

## 2. convert_lerobot3.0_to_2.1.py

**用途**：
将 LeRobot 数据集从 v3.0 版本格式转换回 v2.1 旧格式，便于兼容老版本代码或工具。

**使用方法**：
- 主要通过命令行参数指定数据集路径。
- 运行示例：
  ```bash
  python convert_lerobot3.0_to_2.1.py --repo-id lerobot/pusht --root /path/to/datasets
  ```
- 支持 HuggingFace Hub 数据集快照下载、本地数据集校验、元数据和数据文件批量转换。

**详细案例**：
- 假设你有一个 v3.0 格式的数据集在 `/root/workspace/Embodied_Challenge/lerobot_dataset/cobotmagic_Sim_items_handover_place_000`，转换命令：
  ```bash
  python convert_lerobot3.0_to_2.1.py --repo-id cobotmagic_Sim_items_handover_place_000 --root /root/workspace/Embodied_Challenge/lerobot_dataset/
  ```
- 转换后会在目标目录生成 v2.1 兼容的数据结构和元数据。

---

## 3. OpenPI pi0 on EmbodiChain

This note shows how to run a trained OpenPI pi0 checkpoint as a websocket
policy server and evaluate it in an EmbodiChain simulation environment.

The evaluation script added here is:

```bash
Embodied_Challenge/scripts/eval_openpi0_embodichain.py
```

It is designed for OpenPI checkpoints trained with the local
`pi0_embodichain` or `debug_embodichain` config. Those configs expect runtime
observations with these keys:

```text
observation/state
observation/image
observation/left_wrist_image
observation/right_wrist_image
prompt
```

The script builds those keys from EmbodiChain observations:

```text
robot/qpos                         -> observation/state
sensor/cam_high/color              -> observation/image
sensor/cam_left_wrist/color        -> observation/left_wrist_image
sensor/cam_right_wrist/color       -> observation/right_wrist_image
```

If you want to evaluate an ALOHA-format checkpoint instead, pass
`--policy_input_format aloha`.

### (1). Start OpenPI Server

Run this in one terminal. Replace `CHECKPOINT_DIR` with the trained checkpoint
step directory. In this workspace, the available press-button checkpoint is:

```text
openpi/checkpoints/pi0_embodichain_press_button/26000
```

```bash
cd /home/wu/AA-Program/docker-volume-EmbodiChain/openpi

uv run openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi0_embodichain_press_button --policy.dir=openpi/checkpoints/pi0_embodichain_press_button/26000 --port=8000 --default_prompt="Press the button"

```

### (2). Run EmbodiChain Evaluation

Run this in a second terminal from the workspace root:

```bash
cd /home/wu/AA-Program/docker-volume-EmbodiChain

python Embodied_Challenge/scripts/eval_openpi0_embodichain.py \
  --gym_config Embodied_Challenge/configs/beaker_mixer/gym_config.json \
  --action_config Embodied_Challenge/configs/beaker_mixer/action_config.json \
  --num_envs 1 \
  --device cuda \
  --enable_rt \
  --filter_visual_rand \
  --filter_dataset_saving \
  --host 127.0.0.1 \
  --port 8000 \
  --episodes 20 \
  --max_steps 500 \
  --output results/pi0_beaker_mixer_eval.json
```

You can evaluate another EmbodiChain task by swapping `--gym_config` and
`--action_config`, as long as the policy was trained for the same observation
and action convention.

### (3). Metrics

The output JSON contains one record per episode and a summary block.

Key fields:

```text
success_rate
  successes / episodes, using EmbodiChain's task success signal.

action_steps
  Number of env.step() calls that executed OpenPI actions in each episode.
  This is the per-task action_step count.

model_infer_calls
  Number of OpenPI model chunk requests in each episode. pi0 returns an action
  chunk, so this is usually ceil(action_steps / action_horizon).

model_forward_ms
  Raw server-reported policy_timing.infer_ms values for the episode. Each value
  corresponds to one model.sample_actions() call, not one env step.

episode_mean_model_forward_ms
  Mean of pure model forward time over all model chunk calls. This excludes
  websocket transfer, observation conversion, simulation stepping, action
  clipping and JSON writing.
```


## run_env.py

**用途**：
用于运行 embodied_challenge 环境，支持自定义环境配置、动作配置等。

**使用方法**：
- 支持命令行参数，自动加载环境配置。
- 运行示例：
  ```bash
  python run_env.py --env-id=YourEnvID --config=your_config.yaml
  ```
- 具体参数可通过 `--help` 查看。

**详细案例**：
- 运行 gym 环境并指定配置：
  ```bash
  python run_env.py --env-id=embodied_challenge-v0 --config=configs/beaker_mixer/gym_config_duel.json
  ```
- 支持自定义动作空间、观测空间等高级参数。

---

如需补充更多脚本说明，请补充到本 README。
