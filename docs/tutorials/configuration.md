# Configuration Tutorial

A CobotMagic deployment references one scene and one embodiment.

```yaml
id: RoboSyn-ClickBell-v1
environment:
  component: env.yaml
embodiment:
  component: ../cobotmagic.yaml
collection:
  max_attempts: 20
```

The scene component declares the physics backend, rigid objects, articulations, reset and interval events, and the LeRobot recorder. The embodiment component declares the robot, cameras, control parts and skill resources. Use the `clear` and `random` components to adjust the same task's layout and domain randomization.

For camera randomization, use `eye_range`, `target_range` and `up_range` for a look-at camera, or `pos_range` and `euler_range` for a camera with pose extrinsics.

The task's `task_program/` contains its expert sequence, skill options and physics acceptance wiring. Run the deployment with `launch/run_task.sh <task> <clear|random>`. See [task trajectory authoring](task_trajectory.md) for the program files and [collection](collect_data.md) for dataset paths.
