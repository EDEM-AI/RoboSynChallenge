# RoboSynChallenge/scripts 目录说明

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

## 2. import_lerobot_v21.py

读取本地 LeRobot v2.1 数据并写入新的 v3.0 目录，保留源数据、相机流和动作数组。

```bash
uv run --locked --python 3.12 python -m scripts.import_lerobot_v21 /data/legacy /data/imported-v30
```

---

## 运行任务、回放和分析

```bash
uv run --locked --python 3.12 python -m scripts.run_env \
  --gym_config configs/click_bell/clear/task.cobotmagic.yaml --headless --max_episodes 1
bash launch/run_visualize.sh click_bell clear --resets 1 --output-root /data/visualization
uv run --locked --python 3.12 python -m scripts.analyze_rigid_spawn_range \
  --gym_config configs/sample_loading/random/task.cobotmagic.yaml --list-events
```

仓库内的旧场景 JSON 路径会由兼容入口定位到对应的原生部署。范围分析使用原生专家执行，`--rollout` 同时记录当前任务的物理验收结果。

## 为 LeRobot v3.0 补录末端位姿

```bash
uv run --locked --python 3.12 python -m scripts.add_lerobot_eef_pose \
  --dataset /data/task-dataset --gym_config configs/click_bell/clear/task.cobotmagic.yaml --dry-run
```

默认在新目录写入末端位姿和统计。使用与录制一致的 CobotMagic 部署进行 FK。

## 策略评估

通过 [启动脚本](../launch/README.md) 评估 ACT、Diffusion、SmolVLA 和 OpenPI。评估输出包含成功率、每回合结果和推理采样时间。单独连接 OpenPI WebSocket 服务时，可用 `python -m scripts.eval_openpi0_embodichain --gym_config <部署 YAML>`。
