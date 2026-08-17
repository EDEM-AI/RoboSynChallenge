<div align="center">
<h1>RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills</h1>

<h2 align="center"> 👉<a href="https://edem-ai.github.io/robosynchallenge.github.io/">Webpage</a> | <a href="https://edem-ai.github.io/RoboSynChallenge/html/">Document</a> | <a href="">Paper</a> | <a href="https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard">Leaderboard</a></h2>

![image](misc/robosynchallenge-pipeline.png)

</div>

---

# Contents

- [Contents](#contents)
- [Installtion](#installtion)
- [Datasets](#datasets)
- [Training and Evaluation](#training-and-evaluation)
- [Released Checkpoint Results](#released-checkpoint-results)
- [LeaderBoard](#leaderboard)

# Installtion
Based on the [**EmbodiChain**](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html), we offer both `docker` and `local` installation methods. For detailed installation instructions, please refer to [**Installation Document**](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html).

# Datasets
We provide 1,000 pre-collected trajectories per task as part of the open-source release **RoboSynChallenge** Dataset. The datasets hosted on HuggingFace are available at [here](https://edem-ai.github.io/robosynchallenge.github.io/#/data).

However, we still strongly recommend users to perform data collection themselves. For detailed data collection instructions, please refer to [**Data Collection Document**](https://edem-ai.github.io/RoboSynChallenge/html/tutorials/collect_data.html).


# Training and Evaluation
Currently, RoboSynChallenge integrates training and evaluation for <a href="https://github.com/Physical-Intelligence/openpi">PI0</a>, <a href="https://github.com/Physical-Intelligence/openpi">PI0.5</a>, and <a href="https://github.com/thu-ml/Motus">Motus</a>. Detailed procedures can be found in the documentation for the corresponding strategies: 👉<a href="https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/index.html">Webpage</a>.
In addition, you can easily extend your own policys for training and evaluation by following the documentation 👉<a href="https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html.html">Webpage</a>.

# Released Checkpoint Results

100-episode simulation evaluations for the released ACT and Diffusion Policy checkpoints are published in [`evaluation_results`](evaluation_results/README.md). The machine-readable result file pins every score to its Hugging Face checkpoint revision and `random` protocol configuration.

# LeaderBoard
The full leaderboard and setting can be found in: https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard.