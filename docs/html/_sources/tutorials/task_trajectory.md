# Task Trajectory Implementation

Native EmbodiChain Task Programs specify the expert behavior for RoboSynChallenge.

## 📃 Core Files

- `program.yaml` describes ordered segments, resource roles and semantic calls such as pick, handover and place.
- `integration.yaml` binds those calls to scene entities, calibrated affordances and Atomic Skill options.
- `execution_policy.yaml` configures motion planning and recovery.
- `evaluation.yaml`, when present, declares physical acceptance rules such as insertion distance and uprightness.

### Full Data Flow Pipeline

The deployment loads the environment and robot, binds semantic goals to task objects and robot resources, then calls the native Atomic Skills in program order. The recorder writes the episode and the program's segment annotations.

## 🧠 Standard Step-by-Step Guide

Start with an existing program, such as `configs/sample_loading/task_program/program.yaml`. Program bindings use scene entity IDs and robot resource IDs. The same program runs against whichever CobotMagic deployment is selected at launch.

When a task needs geometry specific to a challenge asset, use the project’s registered semantic lowerers to provide measured targets to the native SDK skills. For example, the test tube program aligns it above the rack, lowers it while keeping the grasp closed, releases it, and waits for it to settle before checking the measured result.

Record a native trajectory and replay it from its seed.

```bash
CUDA_VISIBLE_DEVICES=0 bash launch/run_task.sh click_bell clear \
  --seeded --seed 42 --headless --renderer fast-rt --record_trajectory \
  --trajectory_save_dir datasets/bell-trajectories --dataset-dir datasets/bell
CUDA_VISIBLE_DEVICES=0 bash launch/replay_task.sh click_bell clear \
  datasets/bell-trajectories/traj_env0_000000.pt --seeded-reset --headless --renderer fast-rt
```

## 🔧 Debug Reference

If a native recording’s seed reconstructs a different initial
state, pass `--recorded-reset` to restore the recorded robot and rigid-body
state with EmbodiChain and then replay the controller actions under physics.
Add `--dataset-dir` to record the replay as LeRobot v3.0. The recorder uses
the source segment instructions and commits only a successful full replay.
