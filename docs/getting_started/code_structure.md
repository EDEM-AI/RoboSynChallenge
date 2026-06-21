# Code Structures

Here is the overview of the codebase with some comments.

## Enviornments
```shell
RoboSynChallenge
├── assets # All other required assets besides those provided by EmbodiChain
├── configs # Actions and gym configurations for 10 tasks
├── robosynchallenge
│   ├── data
│   ├── Distractor # Distractor assets and configuration
│   ├── managers # The functor implemented by RoboSynChallenge
│   ├── tasks # The action bank and environment class definitions for 10 tasks.
│   ├── utils
├── launch # Data collection scripts
│   ├── clear # for clear envs
│   ├── random # for envs including domain randomization
│   └── visualize_distribution # for visualizing the range of rigid body randomization in simulation environments
├── scripts
│   ├── analyze_rigid_spawn_range.py # Script for analyzing the feasible range of rigid body position randomization
│   ├── convert_lerobot3.0_to_2.1.py # Script for converting lerobot3.0 to 2.1 format
│   └── eval_openpi0_embodichain.py # Script for openpi eval
└──
```
