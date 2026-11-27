# Code Structures

Here is the overview of the codebase with some comments.

## Enviornments
```shell
RoboSynChallenge
├── assets # All other required assets besides those provided by EmbodiChain
├── configs # Native deployments and Task Programs for 14 tasks
├── robosynchallenge
│   ├── data
│   ├── Distractor # Distractor assets and configuration
│   ├── managers # The functor implemented by RoboSynChallenge
│   ├── deployments.py # Locate native task deployments
│   ├── utils
├── launch # Data collection scripts
│   ├── clear # for clear envs
│   ├── random # for envs including domain randomization
│   └── visualize_distribution # for visualizing the range of rigid body randomization in simulation environments
├── scripts
│   ├── analyze_rigid_spawn_range.py # Script for analyzing the feasible range of rigid body position randomization
│   ├── merge_lerobot_v30.py        # Merge LeRobot v3.0 datasets
│   └── eval_openpi0_embodichain.py # Script for openpi eval
└──
```
