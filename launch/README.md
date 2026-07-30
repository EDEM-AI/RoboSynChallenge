# Launch Scripts

`launch/` 目录包含用于运行 RoboSynChallenge 任务环境的脚本集合，覆盖数据收集、环境检查、可视化等工作流。

## 脚本概览

| 脚本 | 用途 |
|---|---|
| `check_all_envs.sh` | 依次运行全部任务环境，检测是否有 bug |
| `run_task.sh` | 运行单个任务环境，收集专家演示数据 |
| `replay_task.sh` | 以 kinematic、dynamic 或 control 模式回放轨迹 |
| `collect_combined_dataset.sh` | 收集 clear + random 数据集并自动合并 |
| `run_visualize.sh` | 可视化单个任务的域随机化效果 |
| `batch_run_visualize.sh` | 批量可视化所有任务的域随机化效果 |
---

## run_task.sh

最核心的数据收集脚本。启动一个任务环境，生成专家演示轨迹并保存为 LeRobot 格式数据集。

```
用法:
  ./launch/run_task.sh <task_name> <setting> [format] [extra_args...]

参数:
  task_name    任务名称（见下方任务列表）
  setting      random 或 clear（是否启用域随机化）
  format       可选：3_0（默认）或 2_1（自动转换为 LeRobot 2.1）
```

**extra_args 常用选项：**

| 选项 | 说明 |
|---|---|
| `--max_episodes N` | 最多收集 N 个 episode |
| `--headless` | 无头模式运行（不显示窗口） |
| `--viser` | 通过浏览器实时可视化（自动启用无头模式） |
| `--viser-host HOST` | Viser 监听地址（默认 `127.0.0.1`） |
| `--viser-port PORT` | Viser 服务端口（默认 `8080`） |
| `--filter_visual_rand` | 禁用视觉随机化 |
| `--filter_dataset_saving` | 禁用数据集保存（仅运行不存盘） |

**示例：**

```bash
# 收集 click_bell 的 clear 数据（2.1 格式），100 个 episode
./launch/run_task.sh click_bell clear 2_1 --max_episodes 100

# 收集 drawer_open_place 的 random 数据，无头模式
./launch/run_task.sh drawer_open_place random 3_0 --headless

# 在浏览器中查看任务；按终端输出的地址访问
./launch/run_task.sh drawer_open_place random --viser

# 不保存数据，只通过浏览器检查环境
./launch/run_task.sh drawer_open_place clear --viser --filter_dataset_saving

# 仅测试环境能否正常运行（不存盘）
./launch/run_task.sh mixer_operating random 3_0 --filter_dataset_saving --max_episodes 1 --headless
```

数据保存路径为 `lerobot_dataset/<task_name>/`。

---

## replay_task.sh

回放 EmbodiChain 原生轨迹，或旧版 `state/action/reward` 格式的单环境轨迹。

```
用法:
  ./launch/replay_task.sh <task_name> <setting> <trajectory.pt> [extra_args...]
```

**示例：**

```bash
# 通过物理仿真回放 drawer_open_place 动作
./launch/replay_task.sh drawer_open_place random /path/to/state_action.pt --replay_mode dynamic

# 仅恢复机器人运动学状态
./launch/replay_task.sh drawer_open_place clear /path/to/state_action.pt --replay_mode kinematic

# 在浏览器中查看轨迹回放
./launch/replay_task.sh drawer_open_place clear /path/to/trajectory.pt --viser
```

`dynamic` 模式会将记录的 action 重新送入环境，由物理引擎重新计算机器人和场景物体的交互。旧版轨迹不包含抽屉等场景物体的状态，因此要复现物体交互时应使用该模式。

`--viser` 还可搭配 `--viser-host`、`--viser-port`、`--viser-fps`、
`--viser-image-fps`、`--viser-soft-body-fps` 和 `--viser-env-ids` 使用。
远程访问时可将监听地址设为 `0.0.0.0`，但应只向可信网络或客户端开放。

---

## check_all_envs.sh

依次运行全部任务环境（random + clear 两轮），用于快速排查环境是否有加载错误或运行时 bug。

```
用法:
  ./launch/check_all_envs.sh [extra_args...]
```

**行为：**

- 对 10 个正式任务每个执行 random 和 clear 两轮，共 20 次检查
- 每次以 `--headless --filter_dataset_saving --max_episodes 1` 运行，不保存数据
- 失败的任务保留日志到 `/tmp/check_env_<task>_<setting>_*.log`
- 最终打印通过/失败汇总

**示例：**

```bash
# 完整检查（默认）
./launch/check_all_envs.sh

# 每个任务只跑 1 个 episode（默认就是）
./launch/check_all_envs.sh --max_episodes 1
```

**注意：** 排除 `configs/other_tasks/` 下的非正式任务。

---

## run_visualize.sh

运行单个任务的域随机化可视化，循环 reset 环境并渲染，用于检查随机化参数是否合理、物体是否穿模等。

```
用法:
  ./launch/run_visualize.sh <task_name> <setting> [extra_args...]

参数:
  task_name    任务名称
  setting      random 或 clear
```

**extra_args 常用选项：**

| 选项 | 说明 |
|---|---|
| `--resets N` | reset 次数（默认 100） |
| `--headless` | 无头模式（默认已启用） |
| `--viser` | 同时在 Viser 浏览器中查看实时场景 |

**示例：**

```bash
# 可视化 click_bell 的 random 配置，reset 100 次
./launch/run_visualize.sh click_bell random --resets 100

# 可视化 item_assembly 的 clear 配置，reset 50 次
./launch/run_visualize.sh item_assembly clear --resets 50

# 生成分布图时，同时在浏览器中查看实时 reset 过程
./launch/run_visualize.sh item_assembly random --viser
```

该脚本调用 `scripts/visualize_distribution.py`，在窗口中循环 reset 环境，方便肉眼检查场景布局和随机化范围。

---

## batch_run_visualize.sh

批量运行所有任务的可视化，等同于对每个任务依次调用 `run_visualize.sh`。

```
用法:
  ./launch/batch_run_visualize.sh [setting] [extra_args...]

参数:
  setting      random 或 clear（默认 clear）
```

**示例：**

```bash
# 批量可视化所有任务的 clear 配置
./launch/batch_run_visualize.sh

# 批量可视化所有任务的 random 配置，每个 reset 50 次
./launch/batch_run_visualize.sh random --resets 50
```

---

## collect_combined_dataset.sh

自动收集一个任务的 clear 和 random 数据集，并将两个数据集合并为一个。

```
用法:
  ./launch/collect_combined_dataset.sh <task_name> <clear_episodes> <random_episodes> <output_name> [extra_args...]

参数:
  task_name         任务名称
  clear_episodes    clear 模式收集的 episode 数
  random_episodes   random 模式收集的 episode 数
  output_name       合并后的数据集名称
```

**工作流：**

1. 以 clear 模式运行 `run_task.sh`，收集指定数量的 episode
2. 以 random 模式运行 `run_task.sh`，收集指定数量的 episode
3. 自动找到最新生成的两个数据集
4. 调用 `lerobot-edit-dataset` 合并为单一数据集

**示例：**

```bash
# 收集 click_bell 的 50 个 clear + 50 个 random episode，合并为 combined_v1
./launch/collect_combined_dataset.sh click_bell 50 50 combined_v1
```

合并后的数据集位于 `lerobot_dataset/<task_name>/<output_name>/`。

---

## 可用任务列表

### 入门级
- `click_bell` — 按铃
- `handle_basket` — 篮子搬运
- `water_pouring` — 倒水
- `table_rearrangement` — 桌面整理

### 中级
- `items_handover` — 物品交接
- `drawer_open_place` — 开抽屉放置
- `mixer_operating` — 搅拌器操作

### 高级
- `item_assembly` — 物品组装
- `manipulate_pipette` — 移液器操作
- `sample_loading` — 样本加载

---

## 配置文件路径约定

所有脚本遵循统一的配置查找规则：

```
gym_config:    configs/<task_name>/<setting>/gym_config.json
action_config: configs/<task_name>/action_config.json          （优先）
                configs/<task_name>/<setting>/action_config.json（回退）
```

- `random` — 含域随机化（灯光、相机、材质、物体位姿、分心物等）
- `clear` — 不含随机化，固定场景

## policy/pi0/eval.sh

使用 π₀ 策略在 RoboSynChallenge 任务上做推理评估。该脚本位于 `policy/pi0/` 目录下，与策略适配层放在一起。

```
用法:
  ./policy/pi0/eval.sh <task_name> <setting> <train_config> <model_name> <gpu_id> [extra_opts...]

参数:
  task_name       任务名称
  setting         random 或 clear
  train_config    openpi 训练配置名（对应 checkpoints/ 下的目录）
  model_name      模型名称（对应 checkpoints/<train_config>/ 下的目录）
  gpu_id          GPU 编号
```

**extra_opts 示例：**

| 选项 | 说明 |
|---|---|
| `--max_episodes N` | 评估 N 个 episode（默认 100） |
| `--max_steps N` | 每个 episode 最大步数（默认 600） |
| `--pi0_step N` | π₀ 每次推理输出的动作步数（默认 50） |
| `--checkpoint_id N` | checkpoint 步数（默认 30000） |
| `--seed N` | 随机种子 |
| `--viser` | 通过浏览器实时可视化（自动启用无头模式） |

**示例：**

```bash
# 在 click_bell 的 random 配置上评估
./policy/pi0/eval.sh click_bell random my_config pi0_base 0 --checkpoint_id 30000

# 50 个 episode，使用 clear 配置
./policy/pi0/eval.sh water_pouring clear wpm2_embodichain pi0_wpm2 1 --checkpoint_id 10000 --max_episodes 50

# 在浏览器中查看策略评测
./policy/pi0/eval.sh water_pouring clear wpm2_embodichain pi0_wpm2 1 --viser
```

该脚本调用 `scripts/eval_policy.py`，通过 `policy/pi0/deploy_policy.yml` 配置文件统一管理参数。
