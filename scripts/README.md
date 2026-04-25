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

## 3. run_env.py

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
