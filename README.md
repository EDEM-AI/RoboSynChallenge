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
Based on the [**EmbodiChain**](https://github.com/DexForce/EmbodiChain/tree/v0.3.3), we offer both `Docker` and `local uv` installation methods. For detailed installation instructions, please refer to [**Installation Document**](docs/getting_started/installation.md). Both use the root lock for EmbodiChain 0.3.3.

# Datasets
We provide 1,000 pre-collected trajectories per simulation task as part of the open-source release **RoboSynChallenge** Dataset. The LeRobot v3.0 datasets are available in the [Real data](https://huggingface.co/collections/han-xudong/real-data-6b078018b5eb646b7c26fbe1) and [Sim data](https://huggingface.co/collections/han-xudong/sim-data-6b07801c45756c22865cc334) collections on Hugging Face.

However, we still strongly recommend users to perform data collection themselves. For detailed data collection instructions, please refer to [**Data Collection Document**](docs/tutorials/collect_data.md).

# Training and Evaluation
Currently, RoboSynChallenge integrates training and evaluation for [ACT](docs/tutorials/policy/act.md), [Diffusion Policy](docs/tutorials/policy/dp.md), [SmolVLA](docs/tutorials/policy/smolvla.md), [PI0](docs/tutorials/policy/pi0.md), and [PI0.5](docs/tutorials/policy/pi05.md). See the [Motus page](docs/tutorials/policy/motus.md) for the external policy. Detailed procedures can be found in the documentation for the corresponding strategies: 👉<a href="docs/tutorials/policy/index.rst">Webpage</a>.
In addition, you can easily extend your own policys for training and evaluation by following the documentation 👉<a href="docs/tutorials/policy/your_own_policy.md">Webpage</a>.

The [Model checkpoints collection](https://huggingface.co/collections/han-xudong/model-checkpoints-6b07801f9f42e3bccb4a82e0) contains the released ACT, Diffusion and SmolVLA weights.

# Released Checkpoint Results

Historical 100-episode simulation evaluations for the released ACT and Diffusion Policy checkpoints are published in [`evaluation_results`](evaluation_results/README.md). The machine-readable result file pins success rate, action steps, millisecond inference time, Hugging Face checkpoint revisions, and the `random` protocol configuration.

# LeaderBoard
The full leaderboard and setting can be found in: https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard.
