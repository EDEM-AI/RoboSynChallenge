# Task Trajectory Implementation

Native EmbodiChain Task Programs specify the expert behavior for RoboSynChallenge.

- `program.yaml` describes ordered segments, resource roles and semantic calls such as pick, handover and place.
- `integration.yaml` binds those calls to scene entities, calibrated affordances and Atomic Skill options.
- `execution_policy.yaml` configures motion planning and recovery.
- `evaluation.yaml`, when present, declares physical acceptance rules such as insertion distance and uprightness.

Start with an existing program, such as `configs/sample_loading/task_program/program.yaml`. Program bindings use scene entity IDs and robot resource IDs. The same program runs against whichever CobotMagic deployment is selected at launch.

When a task needs geometry specific to a challenge asset, use the project’s registered semantic lowerers to provide measured targets to the native SDK skills. For example, the test tube program aligns it above the rack, lowers it while keeping the grasp closed, releases it, and waits for it to settle before checking the measured result.

Record a native trajectory and replay it from its seed.

```bash
CUDA_VISIBLE_DEVICES=3 bash launch/run_task.sh click_bell clear \
  --seeded --seed 42 --headless --renderer fast-rt --record_trajectory \
  --trajectory_save_dir /data/bell-trajectories --dataset-dir /data/bell
CUDA_VISIBLE_DEVICES=3 bash launch/replay_task.sh click_bell clear \
  /data/bell-trajectories/traj_env0_000000.pt --seeded-reset --headless --renderer fast-rt
```

For a legacy native recording whose seed reconstructs a different initial
state, pass `--recorded-reset` to restore the recorded robot and rigid-body
state with EmbodiChain and then replay the controller actions under physics.
Add `--dataset-dir` to record the replay as LeRobot v3.0. The recorder uses
the source segment instructions and commits only a successful full replay.

Historical task-class imports (for example
`robosynchallenge.tasks.click_bell.ClickBellEnv`) resolve to the native
environment. Select the scene through its deployment YAML. The archived
ActionBank source is available for reference.
