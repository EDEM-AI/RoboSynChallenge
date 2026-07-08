## Task Trajectory Implementation 
> Basic concepts have been covered in prior documents. This manual focuses on underlying implementation principles, standard field writing rules, and function linkage data flow, and disassembles the full trajectory generation logic based on the dual-arm fixture assembly task.

##  📃 Core Files 
1. `action_bank.py`: General underlying operator library
    - Responsibilities: Encapsulate static functions for pose calculation, IK/FK solving, trajectory generation, gripper control, and rigid body binding. Decorators classify node and trajectory functions to expose callable capabilities for cross-task reuse.
    - Loading order: Load and register all functions upon environment startup, serving as the underlying dependency of the configuration file.
2. `action_config.json`: Task-specific workflow graph
    - Responsibilities: Contains no computational logic. It invokes methods inside the bank via string function names, defines control domains, pose generation pipelines, motion trajectories, and multi-arm timing dependencies. Task adjustments only require edits to this file.
    - Loading order: Parsed during environment reset, which drives calls to bank functions to generate all key poses and action sequences.

###  Full Data Flow Pipeline 
Env initialization → Load JSON configuration → Call bank node functions to generate all poses/qpos and store them in the shared cache `env.affordance_datas` → Edges read key points from cache and call trajectory functions to generate time-series actions → Sync rules constrain execution order of multiple actuators → Actions are sent to the simulator for execution.

###  Core Dependency Relationship
`action_config.json` acts as the orchestration and caller layer, while `action_bank.py` serves as the underlying computation implementation layer. The two files exchange data exclusively through the global cache `env.affordance_datas`, and neither can work independently.

##  🏦 Functions in action_bank.py
All capabilities are implemented via static methods decorated with specific tags, which split functions into pose generation nodes and trajectory execution functions.

### 2.1 Core Decorator Principles
1. `@tag_node`
    Marks functions that generate target poses or joint angles. These functions can only be invoked under the `node` field in JSON, and they write new poses/qpos into `affordance_datas` after execution.
2. `@tag_edge`
    Marks functions that generate time-series trajectories or gripper commands. These functions can only be invoked under the `edge` field in JSON and output continuous action arrays.
3. `@resolve_env_params`
    Automatically injects the `env` instance into function parameters, allowing direct read and write access to the shared cache without manual parameter passing.

### 2.2🌰 Example: generate_left_arm_aim_qpos
Underlying logic:
1. Read two 4x4 homogeneous transformation matrices: `guijiao2_pose` and `left_arm_base_pose`.
2. Extract XY translational components and calculate the target horizontal rotation angle of the first joint via `np.arctan2`.
3. Deep copy the initial left arm qpos and overwrite its 0th joint value with the calculated horizontal angle.
4. Store the updated joint angles into the cache entry `left_arm_aim_qpos` for subsequent nodes and trajectories to reference.

Corresponding JSON invocation syntax:
```
{
    "left_arm_aim_qpos": {
        "name": "generate_left_arm_aim_qpos",
        "kwargs": {}
    }
}
```

### 2.3 Gripper Execution
Two separate execution branches are defined based on the flag `return_action`:
- `return_action=false`: Only marks environment state without outputting action sequences.
- `return_action=true`: Outputs a 1-dimensional time-series gripper control sequence.
  - When `expand=true`, `mul_linear_expand` performs linear interpolation to produce smooth opening/closing motion.
  - When `expand=false`, outputs a constant array filled with 0 or 1.
The `duration` parameter controls the total number of simulation steps for the gripper movement.

### 2.4 Core Trajectory Planning
1. `plan_trajectory`
Accepts multiple groups of key joint qpos values and generates smooth joint trajectories via the TOPPRA planner. If the distance between two consecutive key poses is extremely small, the function automatically falls back to `stand_still`.

2. `stand_still`
Generates stationary trajectories by filling all time steps with identical joint angles. If inconsistent key poses are passed, the function retains only the first qpos and prints a warning log.

### 2.5 attach_rigid_objects_now
Creates parent-child binding between two rigid bodies and optionally locks object kinematics to KINEMATIC mode, ensuring synchronized movement of assembled components after connection.

#  🎯 Principles of action_config.json
The configuration file contains five fixed top-level fields: `scope`, `node`, `edge`, `sync`, and `misc`.

## 3.1 scope
Declare all independent actuators (6DoF left/right arms, 1DoF left/right grippers) and define action dimensions, initialization strategies, and data types.

Key field explanations:
- `type`: Fixed value `DiGraph`, all actuators adopt directed graph workflow logic.
- `dim`: Action dimension, set to `[6]` for robotic arms and `[1]` for grippers.
- `init.method`
  - `current_qpos`: Initialize using real-time joint angles from the simulator.
  - `given_qpos`: Initialize with fixed preset joint values.
- `dtype`: Uniformly set to `float32` for all pose and joint calculations.

## 3.2 node

All pose generation nodes use the unified generator `generate_affordances_from_src`, which relies on three core components: `src_key`, `dst_key`, and `pass_processes`.
1. `src_key`: Input cache identifier, referencing the original 4x4 pose matrix or joint angles stored in `affordance_datas`.
2. `dst_key`: Output cache identifier, storing transformed poses or IK-solved joint angles for later nodes and edges to reference.
3. `valid_funcs_name_kwargs_proc`: A pipeline combining validation logic and transformation operators, split into validation functions and sequential `pass_processes` transform operators.

---

🔧 Coordinate system definition
    - `extrinsic`: Global world coordinate system, all translations and rotations are calculated relative to the world origin.
    - `intrinsic`: Local object coordinate system, all transformations are calculated relative to the object’s current self-pose.

🎯 Core operator functions
    - `get_rotation_replaced_pose`: Overwrite the object’s Z-axis rotation with a specified angle.
    - `get_rotated_pose`: Rotate the object around a designated axis by a fixed angle in degrees.
    - `get_offset_pose`: Translate the object along target axes with specified offset values, supporting both intrinsic and extrinsic coordinate modes.
    - `get_offset_qpos`: Directly offset joint values in joint space without Cartesian space conversion.
    - `get_fk_xpos`: Forward kinematics solver, mapping joint qpos to end-effector Cartesian pose.

### 🌰 Example
Complete data flow: Raw object pose → sequential rotation and translation transformations → Cartesian grasp pose `guijiao1_grasp_pose` → IK inverse solving → grasp joint angles `guijiao1_grasp_qpos`.

1. Step 1: Generate grasp pose
```
"src_key": "guijiao1_pose",
"dst_key": "guijiao1_grasp_pose",
"pass_processes": [rotation replacement, 180-degree X-axis rotation, local X-axis offset...]
```
2. Step 2: IK solving to convert pose to qpos
```
"src_key": "guijiao1_grasp_pose",
"dst_key": "guijiao1_grasp_qpos",
"valid_funcs_name_kwargs_proc": [
    {
        "name": "get_ik_ret",
        "kwargs": {
            "ik_func": "env.robot.compute_ik",
            "qpos_seed": "env.affordance_datas['guijiao1_pre1_qpos']",
            "control_part": "right_arm"
        },
        "pass_processes": [{"name": "get_ik_qpos"}]
    },
    {"name": "is_qpos_flip", "kwargs": {...}}
]
```
- `get_ik_ret` and `get_ik_qpos`: Core inverse kinematics solvers. `qpos_seed` provides initial joint guesses to improve solving success rate, and `control_part` specifies the target robotic arm for calculation.
- `is_qpos_flip`: Validates joint flip status to avoid singular joint configurations.

###  Two Categories of Nodes
1. Pose generation nodes: Only apply coordinate transformations without IK solving, output 4x4 homogeneous pose matrices.
2. Qpos generation nodes: Accept pose inputs and output robotic arm joint angles via inverse kinematics solving.

## 3.3 edge
### Core Principle
Directed edges connect two node entries (starting point `src`, target point `sink`). The `name` field binds tag-edge marked functions inside the action bank to generate continuous time-series trajectories.

Key field explanations:
- `src` / `sink`: Cache names of the start and end key nodes.
- `duration`: Total simulation steps allocated for this motion segment; larger values result in slower movement.
- `name`: Bind trajectory functions inside the bank such as `plan_trajectory` or `execute_open`.
- `keypose_names`: Pass start and end node identifiers as key waypoints for trajectory planning.

Two typical edge types:
1. Robotic arm motion edges: Bind `plan_trajectory` to generate smooth interpolated joint trajectories across multiple waypoints.
2. Gripper control edges: Bind `execute_open` or `execute_close` to output 1-dimensional gripper opening/closing action sequences.

## 3.4 sync
### Core Principle
Lock execution order via task dependencies to resolve timing conflicts, such as closing the gripper only after the arm reaches the grasp pose and lifting the object only after full gripper closure.

Key field: `depend_tasks`. The current edge will only start execution after all motion edges listed in this array complete fully.

Sample sequential logic:
```json
"rclose0": {"depend_tasks": ["pre1_to_grasp"]},
"grasp_to_up": {"depend_tasks": ["rclose0"]}
```
Logical flow: The right gripper closes only after the right arm reaches the pre-grasp pose; the lifting motion starts only after gripper closure finishes.

## 3.5 misc
Global debug switches. Disable all visualization options during dataset collection to reduce computational overhead:
- `vis_graph`: Toggle rendering of the directed action graph.
- `vis_gantt`: Toggle generation of timeline Gantt charts.
- `warpping`: Trajectory smoothing switch, must remain enabled at all times.

# 🧠 Standard Step-by-Step Guide 
1. Reuse or extend `action_bank.py` to implement required node and edge underlying functions.
2. Fill out the `scope` section to define dimension and initialization rules for all robotic arms and grippers.
3. Layer the `node` section: write raw object pose transformation nodes first, then IK solving nodes for pre-grasp, grasp, placement and home return qpos.
4. Write all `edge` entries to connect sequential nodes and adjust `duration` values to control movement speed.
5. Configure `sync` dependencies to constrain execution timing between grippers and dual arms.
6. Turn off visualization switches in misc to complete the configuration.

# 🔧 Debug Reference
1. Confusion between intrinsic and extrinsic coordinate systems: Incorrect reference frame for offsets and rotations leads to severe grasp position drift.
2. Improper selection of IK `qpos_seed`: Failed inverse kinematics solving or singular joint configurations.
3. Missing sync dependencies: Chaotic execution order of gripper actions and dual-arm movements.
4. Unreasonable `duration` values: Excessively fast motion causes object penetration; overly slow motion reduces dataset sampling efficiency.

---

#  Summary
1. `action_bank.py`: Underlying computation foundation, providing general functions for pose transformation, trajectory planning and simulator interaction, decoupled from specific assembly tasks.
2. `action_config.json`: Task workflow orchestration layer. It only organizes function calls and timing rules, allowing rapid adaptation to new assembly tasks without modifying underlying code.
3. Core full task generation pipeline: Coordinate transformation operators in nodes generate target Cartesian poses → IK solving nodes convert poses to joint qpos → Edges interpolate continuous motion trajectories → Sync rules enforce multi-actuator timing constraints, finally outputting complete dual-arm assembly action sequences.

For more details, please refer to 👉 [EmbodiChain tutorial Docs]https://dexforce.github.io/EmbodiChain/main/tutorial/index.html
