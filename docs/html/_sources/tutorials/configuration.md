# Configuration Tutorial

## 🔧 Gym Configuration

### 🎯 TaskSettings

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

The scene component declares the physics backend, rigid objects, and articulations. The embodiment component declares the robot, cameras, control parts and skill resources.

### 🧠 Domain Randomization

Use the `clear` and `random` scene components to adjust the same task's layout and domain randomization. Reset and interval events are declared in the selected `env.yaml`.

### 📷 Camera Configuration

For camera randomization, use `eye_range`, `target_range` and `up_range` for a look-at camera, or `pos_range` and `euler_range` for a camera with pose extrinsics.

### 📦 Data Collection Settings

The selected `env.yaml` declares the LeRobot recorder and observation fields. The deployment's `collection` settings control collection attempts. See [Collect Data](collect_data.md) for dataset paths.

## 🔧 Action Configuration

The task's `task_program/` contains its expert sequence, skill options and physics acceptance wiring. Run the deployment with `launch/run_task.sh <task> <clear|random>`. See [task trajectory authoring](task_trajectory.md) for the program files and [collection](collect_data.md) for dataset paths.
