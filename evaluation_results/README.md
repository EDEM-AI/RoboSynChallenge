# Released Checkpoint Evaluation

This directory records the organizer evaluation of the released ACT and Diffusion Policy (DP) simulation checkpoints.

Each checkpoint was evaluated with the task's `random` configuration for 100 episodes. The success rate is therefore also the number of successful episodes out of 100. Average action steps use the recorded task-completion step for successes and the task's `gym_config.json` `max_episode_steps` for every failure. ACT retains the historical success counts and task-limit extrapolation. DP was freshly evaluated with 10 denoising steps, and its action-step and inference metrics come directly from those rollouts. The DP protocol is pinned to commit [`755af40`](https://github.com/EDEM-AI/RoboSynChallenge/commit/755af40a00492f43bd4b9ead68b0def456b7c8b1), and every checkpoint link below is pinned to the evaluated Hugging Face revision.

| Task | ACT success | ACT avg steps | ACT est. inference/episode | DP success (10-step) | DP avg steps | DP avg inference/call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Click bell | [37%](https://huggingface.co/RoboSynChallenge/ACT_sim_click_bell/tree/677e65fbb15974024ff840893496197ef7db26d4) | 259.43/361 | 0.067 s | [51%](https://huggingface.co/RoboSynChallenge/DP_sim_click_bell/tree/fb8c9551e0fade7c4888a926ab577b85f0684da1) | 202.68/361 | 0.142 s |
| Drawer open and place | [29%](https://huggingface.co/RoboSynChallenge/ACT_sim_drawer_open_place/tree/592e30434aad83cf7bd8f1ee105d7c401488743d) | 741.70/900 | 0.174 s | [0%](https://huggingface.co/RoboSynChallenge/DP_sim_drawer_open_place/tree/c5600daa96623adb4f8cd0c156b70836c30b1288) | 900.00/900 | 0.099 s |
| Mixer operating | [77%](https://huggingface.co/RoboSynChallenge/ACT_sim_mixer_operating/tree/0f12c53a2a6e093ae5e1e28f20480296b45fdf2b) | 310.49/500 | 0.075 s | [66%](https://huggingface.co/RoboSynChallenge/DP_sim_mixer_operating/tree/c05afece66ead46b47f6532c86d95cb3dd0f628e) | 340.93/500 | 0.089 s |
| Table rearrangement | [63%](https://huggingface.co/RoboSynChallenge/ACT_sim_table_rearrangement/tree/3a46b36fade1772176b791dd87349f479d0a98c8) | 218.37/361 | 0.058 s | [14%](https://huggingface.co/RoboSynChallenge/DP_sim_table_rearrangement/tree/99c73475a13ec2583b5105dc3773f88bbdeba9f5) | 330.38/361 | 0.174 s |
| Water pouring | [72%](https://huggingface.co/RoboSynChallenge/ACT_sim_water_pouring/tree/0bf0fcfc931a69c52871385f28068fcf873cf07a) | 276.00/500 | 0.066 s | [36%](https://huggingface.co/RoboSynChallenge/DP_sim_water_pouring/tree/b67e1ce7444dfa2940970088ebbe04a8b01013cc) | 385.54/500 | 0.160 s |
| **Five-task macro average** | **55.6%** | **361.20** | **0.088 s** | **33.4%** | **431.91** | **0.124 s** |

The drawer ACT figure comes from the separately recorded `drawer_drive050_grip20` modified-physics run. The DP drawer figure is the fresh 10-step evaluation under the current modified-physics random configuration; neither drawer figure should be presented as an unmodified-physics result. The complete ACT drawer run contains 29 successes, not 30. The table ACT figure is the complete 100-episode 10K-checkpoint run and contains 63 successes, not 70.

The timing boundary includes raw observation lookup and preprocessing, CPU-to-GPU transfer, policy inference, action validation, and GPU-to-environment action transfer; it excludes `env.step()`. The ACT per-episode figures remain estimates derived from an isolated RTX 5090 benchmark. The DP per-call figures are measured directly in the fresh rollouts: each task used four concurrent shards on 4 x RTX 5090 with 56 Intel Xeon Gold 6530 CPU cores total (14 per shard), PyTorch 2.7.1+cu128, CUDA 12.8, and video recording enabled. For comparison, an isolated single-RTX-5090 DP benchmark with the released water-pouring checkpoint, 10 denoising steps, 20 warmups, and 1000 measured calls averaged 54.301 ms per call (52.570 ms median, 55.119 ms p95).

The machine-readable source is [`released_checkpoint_results.json`](released_checkpoint_results.json). It includes the exact checkpoint repositories, revisions, episode counts, action-step metrics, inference benchmark metadata, and protocol paths.

## Reproduce an Evaluation

Use the policy-specific evaluation wrapper with the released checkpoint and the `random` setting:

```bash
bash policy/act/eval.sh <task_name> random <act_checkpoint_path> 0 --headless True
bash policy/dp/eval.sh <task_name> random <dp_checkpoint_path> 0 --headless True
```

The DP wrapper defaults to 10 denoising steps. Pass `--dp_num_inference_steps <N>` after the normal arguments only when an explicit override is required.

The result artifacts are written under `eval_result/<task_name>/<policy>/random/`.
