# Released Checkpoint Evaluation

This directory records the organizer evaluation of the released ACT and Diffusion Policy (DP) simulation checkpoints.

Each checkpoint was evaluated with the task's `random` configuration for 100 episodes. The success rate is therefore also the number of successful episodes out of 100. Average action steps use the recorded task-completion step for successes and the task's `gym_config.json` `max_episode_steps` for every failure. The step metrics and inference totals below are historical extrapolations: they retain the recorded success counts and successful-episode steps, while replacing the former 1000-step failure contribution with the task-specific limit. They are not fresh rollouts under the shorter boundary. The protocol configuration is pinned to commit [`bd6bf77`](https://github.com/EDEM-AI/RoboSynChallenge/commit/bd6bf77a63300f4b9a9d32337b519194dc7311a4), and every checkpoint link below is pinned to the evaluated Hugging Face revision.

| Task | ACT success | ACT avg steps | ACT est. inference/episode | DP success | DP avg steps | DP est. inference/episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Click bell | [37%](https://huggingface.co/RoboSynChallenge/ACT_sim_click_bell/tree/677e65fbb15974024ff840893496197ef7db26d4) | 259.43/361 | 0.067 s | [44%](https://huggingface.co/RoboSynChallenge/DP_sim_click_bell/tree/fb8c9551e0fade7c4888a926ab577b85f0684da1) | 226.80/361 | 3.493 s |
| Drawer open and place | [29%](https://huggingface.co/RoboSynChallenge/ACT_sim_drawer_open_place/tree/592e30434aad83cf7bd8f1ee105d7c401488743d) | 741.70/900 | 0.174 s | [0%](https://huggingface.co/RoboSynChallenge/DP_sim_drawer_open_place/tree/c5600daa96623adb4f8cd0c156b70836c30b1288) | 900.00/900 | 13.275 s |
| Mixer operating | [77%](https://huggingface.co/RoboSynChallenge/ACT_sim_mixer_operating/tree/0f12c53a2a6e093ae5e1e28f20480296b45fdf2b) | 310.49/500 | 0.075 s | [69%](https://huggingface.co/RoboSynChallenge/DP_sim_mixer_operating/tree/c05afece66ead46b47f6532c86d95cb3dd0f628e) | 335.24/500 | 4.990 s |
| Table rearrangement | [63%](https://huggingface.co/RoboSynChallenge/ACT_sim_table_rearrangement/tree/3a46b36fade1772176b791dd87349f479d0a98c8) | 218.37/361 | 0.058 s | [16%](https://huggingface.co/RoboSynChallenge/DP_sim_table_rearrangement/tree/99c73475a13ec2583b5105dc3773f88bbdeba9f5) | 318.84/361 | 4.880 s |
| Water pouring | [72%](https://huggingface.co/RoboSynChallenge/ACT_sim_water_pouring/tree/0bf0fcfc931a69c52871385f28068fcf873cf07a) | 276.00/500 | 0.066 s | [33%](https://huggingface.co/RoboSynChallenge/DP_sim_water_pouring/tree/b67e1ce7444dfa2940970088ebbe04a8b01013cc) | 396.76/500 | 5.818 s |
| **Five-task macro average** | **55.6%** | **361.20** | **0.088 s** | **32.4%** | **435.53** | **6.491 s** |

The drawer figures come from the separately recorded modified-physics runs (`drawer_drive050_grip20` for ACT and `dp_drawer_pr31_d50g3` for DP); they must not be presented as unmodified-protocol results. The complete ACT drawer run contains 29 successes, not 30. The table ACT figure is the complete 100-episode 10K-checkpoint run and contains 63 successes, not 70.

Inference latency was measured with the released water-pouring checkpoint for each policy on an NVIDIA GeForce RTX 5090 (driver 580.95.05, PyTorch 2.7.1+cu128, CUDA 12.8) with 14 Intel Xeon Gold 6530 CPU cores assigned to each independent job. Synthetic deployment-shaped raw observations used batch size 1, one 14D CPU state, and three 480x640 RGBA CPU images. The synchronized timing boundary includes raw observation lookup and preprocessing, CPU-to-GPU transfer, policy inference, action validation, and GPU-to-environment action transfer; it excludes `env.step()`. After 10 warmup calls, ACT averaged 11.608 ms over 200 calls (10 timeout-episode equivalents at horizon 50), and DP averaged 457.760 ms over 320 calls (10 timeout-episode equivalents at horizon 32). Per-task inference time is an estimate: the measured per-call latency is multiplied by the exact average number of model calls in each historical rollout.

The machine-readable source is [`released_checkpoint_results.json`](released_checkpoint_results.json). It includes the exact checkpoint repositories, revisions, episode counts, action-step metrics, inference benchmark metadata, and protocol paths.

## Reproduce an Evaluation

Use the policy-specific evaluation wrapper with the released checkpoint and the `random` setting:

```bash
bash policy/act/eval.sh <task_name> random <act_checkpoint_path> 0 --headless True
bash policy/dp/eval.sh <task_name> random <dp_checkpoint_path> 0 --headless True
```

The result artifacts are written under `eval_result/<task_name>/<policy>/random/`.
