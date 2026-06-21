# Configuration Tutorial

All configuration files are stored in the `configs` folder and follow the standard json format. Each task contains its own configuration folder, which includes `action_config.json` and `gym_config.json`. For a more detailed guide on writing configurations to implement a task env, please refer to [EmbodiChain Data Generation](https://dexforce.github.io/EmbodiChain/main/tutorial/data_generation.html#). Only a brief introduction is provided here.

---

## 🔧 Gym Configuration

### 🎯 TaskSettings

```json
{
    "id": "BeakerMixer-v0",
    "max_episodes": 100,
    "max_episode_steps": 500
}
```
| Field          | Type | Required | Description                                                                                                                                               |
|----------------|------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`  | string  | ✅        | The ID here must be exactly the same as the registered environment name, for example `@register_env("BeakerMixer-v0")` in the `RoboSynChallenge/robsynchallenge/tasks/beaker_mixer/beaker_mixer.py`                                                                                                           |
| `max_episodes`  | int  | ✅        | Number of **successful episodes** to collect.                                                                                                             |
| `max_episode_steps` | int  | ✅        | Maximum step supported by the environment.                                            |

### 🧠 Domain Randomization
Configure task variation for better generalization. Most event functors can be found in detail in: 👉 [EmbodiChain Randomization Docs](https://dexforce.github.io/EmbodiChain/main/api_reference/embodichain/embodichain.lab.gym.envs.managers.html#module-embodichain.lab.gym.envs.managers.randomization).

```json
{
"random_light": {
    "func": "randomize_light",
    "mode": "interval",
    "interval_step": 10,
    "params": {
        "entity_cfg": {"uid": "light_0"},
        "position_range": [[-0.5, -0.5, 2], [0.5, 0.5, 2]],
        "color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
        "intensity_range": [10.0, 30.0]
    }
},
"random_camera_high_intrinsics": {
    "func": "randomize_camera_intrinsics",
    "mode": "reset",
    "params": {
        "entity_cfg": {"uid": "cam_high"},
        "focal_x_range": [-50, 50],
        "focal_y_range": [-50, 50]
    }
},
"random_camera_high_extrinsics": {
    "func": "randomize_camera_extrinsics",
    "mode": "reset",
    "params": {
        "entity_cfg": {"uid": "cam_high"},
        "pos_range": [[0.0, -0.02, 0.0], [0.02, 0.02, 0.02]],
        "euler_range": [[-0.175, -0.175, -0.175], [0.175, 0.175, 0.175]]
    }
},
"random_robot_init_eef_pose": {
    "func": "randomize_robot_eef_pose",
    "mode": "reset",
    "params": {
        "entity_cfg": {"uid": "CobotMagic", "control_parts": ["left_arm", "right_arm"]},
        "position_range":  [[-0.01, -0.01, -0.01], [0.01, 0.01, 0]]
    }
},
"random_robot_qpos": {
    "func": "randomize_robot_qpos",
    "mode": "reset",
    "params": {
        "entity_cfg": {"uid": "CobotMagic"},
        "qpos_range": [
            [-0.06, -0.05, -0.06, -0.06, -0.05, -0.05, -0.06, -0.05, -0.06, -0.06, -0.05, -0.05],
            [0.06, 0.05, 0.06, 0.06, 0.05, 0.05, 0.06, 0.05, 0.06, 0.06, 0.05, 0.05]
        ],
        "relative_qpos": true,
        "joint_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }
},
"random_material": {
    "func": "randomize_visual_material",
    "mode": "interval",
    "interval_step": 10,
    "params": {
        "entity_cfg": {"uid": "table"},
        "random_texture_prob": 0.5,
        "texture_path": "/root/workspace/RoboSynChallenge/assets/background_texture/100",
        "base_color_range": [[0.2, 0.2, 0.2], [1.0, 1.0, 1.0]]
    }
},
"random_robot_material": {
    "func": "randomize_visual_material",
    "mode": "interval",
    "interval_step": 10,
    "params": {
        "entity_cfg": {"uid": "CobotMagic", "link_names": [".*"]},
        "random_texture_prob": 0.5,
        "texture_path": "/root/workspace/RoboSynChallenge/assets/background_texture/100",
        "base_color_range": [[0.2, 0.2, 0.2], [1.0, 1.0, 1.0]]
    }
},
"random_plane_material": {
    "func": "randomize_visual_material",
    "mode": "interval",
    "interval_step": 10,
    "params": {
        "entity_cfg": {"uid": "default_plane"},
        "random_texture_prob": 0.5,
        "texture_path": "/root/workspace/RoboSynChallenge/assets/background_texture/100",
        "base_color_range": [[0.2, 0.2, 0.2], [1.0, 1.0, 1.0]]
    }
},
"set_beaker_material": {
    "func": "set_rigid_object_visual_material",
    "mode": "startup",
    "params": {
        "entity_cfg": {"uid": "beaker"},
        "mat_cfg": {
            "uid": "beaker_mat",
            "base_color": [1.0, 1.0, 0.999, 1.0],
            "ior": 1.47,
            "roughness": 0.005,
            "metallic": 0.0,
            "material_type": "BSDF"
        }
    }
},
"random_beaker_mixer_material": {
    "func": "randomize_visual_material",
    "mode": "interval",
    "interval_step": 10,
    "params": {
        "entity_cfg": {"uid": "beaker_mixer"},
        "random_texture_prob": 0.5,
        "texture_path": "/root/workspace/RoboSynChallenge/assets/background_texture/100",
        "base_color_range": [[0.2, 0.2, 0.2], [1.0, 1.0, 1.0]]
    }
},
"random_beaker_mixer_pose": {
    "func": "randomize_rigid_object_pose",
    "mode": "reset",
    "params": {
        "entity_cfg": {"uid": "beaker_mixer"},
        "position_range": [[0.54, 0.0, 0.9], [0.65, 0.1, 0.9]],
        "rotation_range": [[0, 0, 0], [0, 0, 0]],
        "relative_position": false,
        "relative_rotation": true
    }
},
"random_beaker_pose": {
    "func": "randomize_rigid_object_pose",
    "mode": "reset",
    "params": {
        "entity_cfg": {"uid": "beaker"},
        "position_range": [[0.63, -0.25 ,0.85], [0.75, -0.2, 0.85]],
        "rotation_range": [[0, 0, -180], [0, 0, 180]],
        "relative_position": false,
        "relative_rotation": true
    }
},
"randomize_distractor_slots": {
    "func": "replace_distractor_slots_from_library",
    "mode": "reset",
    "params": {
        "entity_cfgs": [
            {"uid": "distractor_0"},
            {"uid": "distractor_1"}
        ],
        "library_index_path": "/root/workspace/RoboSynChallenge/robosynchallenge/Distractor/ASSET_INDEX.json",
        "categories": ["PaperCup", "fork", "spoon"],
        "position_ranges": [
            [[0.4, -0.35, 0.89], [1.0, 0.35, 0.92]],
            [[0.4, -0.35, 0.89], [1.0, 0.35, 0.92]]
        ],
        "appear_probs": [1.0, 1.0],
        "z_rotation_ranges": [
            [-180.0, 180.0],
            [-180.0, 180.0]
        ],
        "hide_position": [0.0, 0.0, -10.0],
        "avoid_uids": ["beaker", "beaker_mixer"],
        "min_distance_to_avoid": 0.4,
        "max_resample_attempts": 100,
        "physics_update_step": 1
    }
}
}
```

### 📷 Camera Configuration
Camera configuration can be found in detail in: 👉 [EmbodiChain Camera Docs](https://dexforce.github.io/EmbodiChain/main/api_reference/embodichain/embodichain.lab.sim.sensors.html#embodichain.lab.sim.sensors.Camera).
```json
{
"sensor": [
    {
        "sensor_type": "Camera",
        "uid": "cam_high",
        "width": 640,
        "height": 480,
        "enable_mask": false,
        "intrinsics": [606.315186, 606.100952, 320.549316, 245.877106],
        "extrinsics": {
            "eye": [
                0.257046,
                0.049382,
                1.459689
            ],
            "target": [
                0.599275,
                0.044517,
                0.520085
            ],
            "up": [
                0.939547,
                -0.010356,
                0.342262
            ]
        }
    },
    {
        "sensor_type": "Camera",
        "uid": "cam_right_wrist",
        "width": 640,
        "height": 480,
        "enable_mask": false,
        "intrinsics": [603.117004394531, 602.418640136719, 326.727478027344, 251.933990478516],
        "extrinsics": {
            "parent": "right_link6",
            "pos": [-0.12, 0.01, 0.03],
            "quat": [0.1949, 0.6797, -0.6797, -0.1949]
        }
    },
    {
        "sensor_type": "Camera",
        "uid": "cam_left_wrist",
        "width": 640,
        "height": 480,
        "enable_mask": false,
        "intrinsics": [603.117004394531, 602.418640136719, 326.727478027344, 251.933990478516],
        "extrinsics": {
            "parent": "left_link6",
            "pos": [-0.12, 0.01, 0.03],
            "quat": [0.1949, 0.6797, -0.6797, -0.1949]
        }
    }
]
}
```


### 📦 Data Collection Settings
Dataset configuration can be found in detail in: 👉 [EmbodiChain Data Recorder Docs](https://dexforce.github.io/EmbodiChain/main/overview/gym/dataset_functors.html#lerobotrecorder).

```json
{
"dataset": {
    "lerobot": {
        "func": "LeRobotRecorder",
        "mode": "save",
        "params": {
            "save_path": "/root/workspace/RoboSynChallenge/lerobot_dataset/beaker_mixer_dual",
            "robot_meta": {
                "robot_type": "CobotMagic",
                "control_freq": 25,
                "control_parts": ["left_arm", "left_eef", "right_arm", "right_eef"]
            },
            "instruction": {
                "lang": "Pick the beaker, place it on the mixer, then flip the toggle switch with the other arm."
            },
            "extra": {
                "scene_type": "Sim",
                "task_description": "beaker_mixer_dual",
                "data_type": "sim"
            },
            "use_videos": true
        }
    }
}
}
```
## 🔧 Action Configuration
Action Configuration can be found in detail in: 👉 [EmbodiChain Action Config Docs](https://dexforce.github.io/EmbodiChain/main/tutorial/data_generation.html#step-2-prepare-the-action-configuration)
