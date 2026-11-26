# 启动脚本

从仓库根目录运行。所有命令使用同一个由 `uv.lock` 锁定的 Python 3.12 环境。

```bash
uv sync --locked --python 3.12
```

## 仿真、专家采集与回放

```bash
CUDA_VISIBLE_DEVICES=3 bash launch/run_task.sh click_bell clear \
  --headless --renderer fast-rt --max_episodes 1 --record_trajectory \
  --dataset-dir /data/bell-data --trajectory_save_dir /data/bell-trajectories
CUDA_VISIBLE_DEVICES=3 bash launch/replay_task.sh click_bell clear \
  /data/bell-trajectories/traj_env0_000000.pt --headless --replay_mode dynamic
bash launch/collect_combined_dataset.sh click_bell 3 3 /data/bell-combined --headless
```

采集和合并写入 LeRobot v3.0。普通动态回放由 EmbodiChain 提供。如果需要用轨迹中原始种子初始化场景，将 `--seeded-reset` 作为轨迹路径后的第一个参数。固定种子采集时，将 `--seeded` 放在场景参数后，并传入 `--seed`。可选的 `--affordance-branch` 选择 SDK 的抓取候选。要把动态回放再录成数据集，可使用以下命令。录制使用源轨迹的分段和语言标注，并在物理验收成功后保存。若旧轨迹的重置种子无法重建录制时的起始状态，改用首个参数 `--recorded-reset`，由 EmbodiChain 恢复轨迹中的起始状态后进行动态回放。

验收新采集时，检查 `meta/embodichain_episodes.jsonl` 中的 `success`、`terminal_reason` 和 episode UUID，再用同次采集的轨迹做动态复播。`--seeded-reset` 会核对初始关节和物体状态，逐步执行原控制命令。输出 `Success=True` 要求完整执行且满足任务的物理成功条件。

```bash
CUDA_VISIBLE_DEVICES=3 bash launch/replay_task.sh other_tasks/manipulate_pipette_two_beaker random \
  /data/pipette-trajectories/traj_env0_000000.pt --seeded-reset \
  --headless --renderer fast-rt --dataset-dir /data/pipette-replay
```

## 任务和场景

正式任务包括 `click_bell`、`drawer_open_place`、`handle_basket`、`item_assembly`、`items_handover`、`manipulate_pipette`、`mixer_operating`、`sample_loading`、`table_rearrangement` 和 `water_pouring`。

四个附加任务的标识为 `other_tasks/open_pan`、`other_tasks/manipulate_pipette_two_beaker`、`other_tasks/pour_water` 和 `other_tasks/sample_loading`。其中 `other_tasks/sample_loading` 是右臂直接插管。两组任务都提供 `clear` 和 `random` 场景。

`configs/<任务>/<场景>/task.cobotmagic.yaml` 是入口。该文件组合场景、CobotMagic 机器人和相机。专家步骤与技能参数在任务的 `task_program/`。 `launch/check_all_envs.sh` 依次检查 28 个场景。

## 域随机化可视化

```bash
CUDA_VISIBLE_DEVICES=3 bash launch/run_visualize.sh click_bell random \
  --resets 10 --max-resets 50 --output-root /data/spawn-overlays
```

工具保存专家动作成功执行的场景的初始位置叠图、原始首帧、分割掩码及各采样种子的验收记录。 `launch/batch_run_visualize.sh random` 批量执行。

## 策略

ACT、Diffusion 和 SmolVLA 使用 `launch/train_policy.sh <act|diffusion|smolvla> <任务名|本地目录|Hub ID> <输出>` 和 `launch/eval_policy.sh <act|dp|smolvla> <任务> <场景> [checkpoint] [GPU]`。训练指定任务名时使用 `han-xudong` 下的仿真数据，Hub 数据固定读取 `v3.0`。省略 checkpoint 时，评估加载对应任务的发布权重。参见[数据和权重](../docs/tutorials/download_data.md)。

π₀ 和 π₀.₅ 使用 `launch/openpi.sh` 或各自策略目录的 `finetune.sh` 和 `eval.sh`，训练和续训采用 OpenPI 的 Orbax 格式。RoboSynChallenge 训练配置默认使用 `han-xudong/cobotmagic_Sim_click_bell` 的 `v3.0` 数据。参见 [π₀](../docs/tutorials/policy/pi0.md) 和 [π₀.₅](../docs/tutorials/policy/pi05.md)。

使用 `launch/openpi.sh <pi0|pi05> serve` 启动原生 OpenPI 服务，训练、续训及服务均使用根环境。
