# Installation

Use the repository root on Linux x86-64 with an NVIDIA driver that supports CUDA 12.8. The root project installs Python 3.12, EmbodiChain 0.3.3, DexSim Engine 0.6.0, LeRobot 0.6.1, and the OpenPI JAX dependencies.

```bash
uv sync --locked --python 3.12
uv run --locked --python 3.12 python -c 'from importlib.metadata import version; print(*(version(p) for p in ("embodichain", "dexsim-engine", "lerobot", "jax")))'
```

The same environment runs the simulator, native expert collection, replay, LeRobot ACT/Diffusion/SmolVLA training, and OpenPI π₀/π₀.₅ training. The root lock also selects CUDA packages for both PyTorch and JAX.

## Verify simulation and collection

```bash
CUDA_VISIBLE_DEVICES=3 bash launch/run_task.sh click_bell clear \
  --headless --renderer fast-rt --max_episodes 1 --dataset-dir /data/click-bell
```

Launchers default to the checkout's `assets/` as the EmbodiChain data root. Set `EMBODICHAIN_DATA_ROOT` if those resources are stored elsewhere. For headless GPU rendering, use an environment with the NVIDIA graphics driver libraries exposed.

All collection produces LeRobot v3.0. To import an older local release, use the [dataset importer](../tutorials/collect_data.md). See [task deployments](../tutorials/configuration.md), [expert replay](../tutorials/task_trajectory.md), and [storage](../tutorials/download_data.md).
