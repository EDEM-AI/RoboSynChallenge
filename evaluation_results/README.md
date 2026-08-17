# Released Checkpoint Evaluation

This directory records the organizer evaluation of the released ACT and Diffusion Policy (DP) simulation checkpoints.

Each checkpoint was evaluated with the task's `random` configuration for 100 episodes. The success rate is therefore also the number of successful episodes out of 100. The protocol configuration is pinned to commit [`bd6bf77`](https://github.com/EDEM-AI/RoboSynChallenge/commit/bd6bf77a63300f4b9a9d32337b519194dc7311a4), and every checkpoint link below is pinned to the evaluated Hugging Face revision.

| Task | ACT | DP |
| --- | ---: | ---: |
| Click bell | [37%](https://huggingface.co/RoboSynChallenge/ACT_sim_click_bell/tree/677e65fbb15974024ff840893496197ef7db26d4) | [44%](https://huggingface.co/RoboSynChallenge/DP_sim_click_bell/tree/fb8c9551e0fade7c4888a926ab577b85f0684da1) |
| Drawer open and place | [30%](https://huggingface.co/RoboSynChallenge/ACT_sim_drawer_open_place/tree/592e30434aad83cf7bd8f1ee105d7c401488743d) | [0%](https://huggingface.co/RoboSynChallenge/DP_sim_drawer_open_place/tree/c5600daa96623adb4f8cd0c156b70836c30b1288) |
| Mixer operating | [77%](https://huggingface.co/RoboSynChallenge/ACT_sim_mixer_operating/tree/0f12c53a2a6e093ae5e1e28f20480296b45fdf2b) | [69%](https://huggingface.co/RoboSynChallenge/DP_sim_mixer_operating/tree/c05afece66ead46b47f6532c86d95cb3dd0f628e) |
| Table rearrangement | [70%](https://huggingface.co/RoboSynChallenge/ACT_sim_table_rearrangement/tree/3a46b36fade1772176b791dd87349f479d0a98c8) | [16%](https://huggingface.co/RoboSynChallenge/DP_sim_table_rearrangement/tree/99c73475a13ec2583b5105dc3773f88bbdeba9f5) |
| Water pouring | [72%](https://huggingface.co/RoboSynChallenge/ACT_sim_water_pouring/tree/0bf0fcfc931a69c52871385f28068fcf873cf07a) | [33%](https://huggingface.co/RoboSynChallenge/DP_sim_water_pouring/tree/b67e1ce7444dfa2940970088ebbe04a8b01013cc) |
| **Five-task macro average** | **57.2%** | **32.4%** |

The machine-readable source is [`released_checkpoint_results.json`](released_checkpoint_results.json). It includes the exact checkpoint repositories, revisions, episode counts, maximum episode lengths, and protocol paths.

## Reproduce an Evaluation

Use the policy-specific evaluation wrapper with the released checkpoint and the `random` setting:

```bash
bash policy/act/eval.sh <task_name> random <act_checkpoint_path> 0 --headless True
bash policy/dp/eval.sh <task_name> random <dp_checkpoint_path> 0 --headless True
```

The result artifacts are written under `eval_result/<task_name>/<policy>/random/`.
