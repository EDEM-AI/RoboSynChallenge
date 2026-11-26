<div align="center">
<h1>RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills</h1>

<h2 align="center"> 👉<a href="https://edem-ai.github.io/robosynchallenge.github.io/">Webpage</a> | <a href="https://edem-ai.github.io/RoboSynChallenge/html/">Document</a> | <a href="">Paper</a> | <a href="https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard">Leaderboard</a></h2>

![image](misc/robosynchallenge-pipeline.png)

</div>

---

# Contents

- [Contents](#contents)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training and Evaluation](#training-and-evaluation)
- [Released Checkpoint Results](#released-checkpoint-results)
- [LeaderBoard](#leaderboard)

# Installation
Install the shared Linux GPU environment from the repository root with `uv sync --locked --python 3.12`. See the [installation guide](docs/getting_started/installation.md) for simulation and policy dependencies.

# Datasets
The public RoboSynChallenge datasets use LeRobot v3.0. Browse the [real data](https://huggingface.co/collections/han-xudong/real-data-6b078018b5eb646b7c26fbe1) and [simulation data](https://huggingface.co/collections/han-xudong/sim-data-6b07801c45756c22865cc334) collections.

The current simulator records LeRobot v3.0. See [data collection](docs/tutorials/collect_data.md) and [data, checkpoint and asset storage](docs/tutorials/download_data.md).


# Training and Evaluation
The root environment supports [ACT](docs/tutorials/policy/act.md),
[Diffusion Policy](docs/tutorials/policy/dp.md),
[SmolVLA](docs/tutorials/policy/smolvla.md),
[π₀](docs/tutorials/policy/pi0.md), and [π₀.₅](docs/tutorials/policy/pi05.md).
See the [launcher guide](launch/README.md) for training, evaluation, and
native task replay. To add a policy, start with the
[policy template](docs/tutorials/policy/your_own_policy.md).
The ACT, Diffusion, and SmolVLA evaluation launchers can load the published
checkpoint for the selected task from the [model checkpoints collection](https://huggingface.co/collections/han-xudong/model-checkpoints-6b07801f9f42e3bccb4a82e0).

# Released Checkpoint Results

Historical 100-episode simulation evaluations for the released ACT and Diffusion Policy checkpoints are published in [`evaluation_results`](evaluation_results/README.md). The machine-readable result file pins success rate, action steps, millisecond inference time, Hugging Face checkpoint revisions, and the `random` protocol configuration.

# LeaderBoard
The full leaderboard and setting can be found in: https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard.
