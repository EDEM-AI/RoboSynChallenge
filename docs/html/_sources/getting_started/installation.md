# Installation

Our simulation environment depends on and is built on top of [**EmbodiChain**](https://dexforce.github.io/EmbodiChain/), so we need to install `EmbodiChain` first, and then install the `RobosynChallenge` extension.

## Choose your setup

Based on the [**EmbodiChain installation guide**](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html), we offer two installation methods:

| Path | Best for | Notes |
|------|----------|-------|
| **Docker** | First run, reproducible GPU sim | Pre-built image with CUDA 12.8, Vulkan, and Python 3.11 |
| **Local + [uv](https://github.com/astral-sh/uv)** | Day-to-day development | Fast installs; recommended with a virtual environment |

## Docker (Recommended)

### Prerequisites
- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver ≥ 535 on the host
- For **GUI** runs: working X11 forwarding (`DISPLAY`, `~/.Xauthority`, `/tmp/.X11-unix`)
- For **headless** servers: no display required; use `--headless` in scripts

### 1. Construct docker container

First, please install [**EmbodiChain**](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html) v0.2.3, which is the underlying simulation environment we use.

```bash
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8
# The pre-configured image includes CUDA 12.8, Vulkan-related mounts, and dependencies needed for GPU simulation and rendering.
```

Then, open the container using the image and mount the local workspace.
```bash
mkdir RoboSynChallenge_ws && cd RoboSynChallenge_ws
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
git checkout tags/v0.2.3
./docker/docker_run.sh <container_name> <data_path>
# This will mount <data_path> to the /root/workspace directory in <container_name> container.
# We recommend setting <data_path> to the RoboSynChallenge_ws/ directory and placing EmbodiChain and RoboSynChallenge there.
```

### 2. Install EmbodiChain

After enter the docker container, install the `EmbodiChain` package:
```bash
conda create -n robosyn python=3.11
conda activate robosyn
cd /root/workspace/EmbodiChain
pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
pip install "numpy<2.0"
```

### 3. Install RoboSynChallenge

Then install the `RoboSynChallenge` package:
```bash
cd /root/worspace
git clone https://github.com/EDEM-AI/RoboSynChallenge.git
cd RoboSynChallenge
pip install -e .
```


## Local installation

Use a dedicated virtual environment to avoid conflicts with system Python packages.

### 1. Create a virtual environment

**With uv (recommended):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 2. Install EmbodiChain

**editable install from source with uv:**

```bash
mkdir RoboSynChallenge_ws && cd RoboSynChallenge_ws
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
git checkout tags/v0.2.3
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e . \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site
```

This pulls in `dexsim_engine` (Python package `dexsim`) and the rest of the core dependencies declared in `pyproject.toml`.

### 3. Install RoboSynChallenge

Then install the `RoboSynChallenge` package:

```bash
cd /path/to/RoboSynChallenge_ws
git clone https://github.com/EDEM-AI/RoboSynChallenge.git
cd RoboSynChallenge
uv pip install -e .
```

<!-- todo: embodichain debug -->

## Verify installation

### 1. Verify-Embodichain

```bash
python -c "import embodichain, dexsim; print('embodichain', embodichain.__version__); print('dexsim', dexsim.__version__)"
```
You should see version strings for both packages with no import errors.

The tutorial script `scripts/tutorials/sim/create_scene.py` ships with the repository. Run it from the **repository root**:
```bash
cd /path/to/EmbodiChain
source .venv/bin/activate
python scripts/tutorials/sim/create_scene.py
```

- **With a display:** omit `--headless` to open the DexSim viewer after the scene is built.
- **Headless / SSH:** use `--headless` to run without a window (FPS logs in the terminal):

```bash
python scripts/tutorials/sim/create_scene.py --headless
```
Press `Ctrl+C` to stop; the script cleans up the simulation on exit.

### 2 Verify-RoboSynChallenge

```bash
python -c "import robosynchallenge; print('robosynchallenge', robosynchallenge.__version__)"
```
You should see version strings for this package with no import errors.

You can also execute the following script to verify that the data acquisition function is working properly:
```bash
cd /path/to/RoboSynChallenge_ws/RoboSynChallenge
bash launch/run_task.sh click_bell random 2_1 --max_episodes 1
```
If the installation is successful, the script will start the simulation environment and display the UI of the click_bell environment. Simultaneously, it will collect one episode of data in lerobot format (version 2.1) and save it to `/path/to/RoboSynChallenge/lerobot_dataset/click_bell/`.

The data storage directory can be modified under the corresponding task and the corresponding gym_config settings. For example, the path to the gym_config file for the click_bell task and adding domain randomization is: `/path/to/RoboSynChallenge/configs/click_bell/random/gym_config.json`
