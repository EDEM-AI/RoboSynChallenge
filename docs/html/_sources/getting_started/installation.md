# Installation

Our simulation environment depends on and is built on top of [**EmbodiChain**](https://dexforce.github.io/EmbodiChain/). RoboSynChallenge uses EmbodiChain 0.3.3, DexSim Engine 0.6.0, LeRobot 0.6.1, and Python 3.12. The root `uv.lock` is shared by both installation methods.

## Choose your setup

Based on the [**EmbodiChain installation guide**](https://github.com/DexForce/EmbodiChain/tree/v0.3.3), we offer two installation methods.

| Path | Best for | Notes |
|------|----------|-------|
| **Docker** | First run, reproducible GPU sim | Pre-built image with CUDA 12.8, Vulkan, and Python 3.12 |
| **Local + [uv](https://github.com/astral-sh/uv)** | Day-to-day development | Fast installs with a dedicated virtual environment |

## Docker (Recommended)

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- An NVIDIA host driver that supports CUDA 12.8
- For **GUI** runs, working X11 forwarding (`DISPLAY`, `~/.Xauthority`, `/tmp/.X11-unix`)
- For **headless** servers, use `--headless` in the launch scripts

### 1. Construct docker container

The pre-configured EmbodiChain image includes CUDA 12.8, Vulkan-related mounts, and dependencies needed for GPU simulation and rendering. From the RoboSynChallenge checkout, pull the Python 3.12 image and start the container using the [EmbodiChain 0.3.3 launcher](https://github.com/DexForce/EmbodiChain/blob/v0.3.3/docker/docker_run.sh).

```bash
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8-py312
git clone --branch v0.3.3 --depth 1 https://github.com/DexForce/EmbodiChain.git ../EmbodiChain
bash ../EmbodiChain/docker/docker_run.sh robosynchallenge "$PWD"
```

The launcher mounts the RoboSynChallenge checkout at `/root/workspace` inside the container. An existing EmbodiChain v0.3.3 checkout can also supply the launcher.

### 2. Install RoboSynChallenge and its dependencies

Enter the container and use the root lock to install the project and its dependencies. The container's virtual environment uses a separate path so the mounted checkout can also keep its local `.venv`.

```bash
docker exec -it -e UV_PROJECT_ENVIRONMENT=/opt/robosynchallenge-venv robosynchallenge bash
cd /root/workspace
uv sync --locked --python 3.12
```

Run the verification and RoboSynChallenge launch commands below in this shell.

## Local installation

Use a dedicated virtual environment to avoid conflicts with system Python packages. On Linux x86-64, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the following from the RoboSynChallenge checkout.

```bash
uv sync --locked --python 3.12
```

This pulls in EmbodiChain, DexSim Engine and the policy dependencies declared in the root `pyproject.toml`. For GPU rendering, the system needs the NVIDIA graphics driver libraries.

## Verify installation

### 1. Verify EmbodiChain

```bash
uv run --locked --python 3.12 python -c 'from importlib.metadata import version; print(*(version(p) for p in ("embodichain", "dexsim-engine", "lerobot")))'
```

You should see version strings `0.3.3`, `0.6.0` and `0.6.1`.

### 2. Verify RoboSynChallenge

You can also execute the following script to verify that the data acquisition function is working properly.

```bash
bash launch/run_task.sh click_bell random --headless --renderer fast-rt --max_episodes 1
```

The script collects one episode in LeRobot v3.0. See [Collect Data](../tutorials/collect_data.md) for output locations and [Configuration Tutorial](../tutorials/configuration.md) for task settings.
