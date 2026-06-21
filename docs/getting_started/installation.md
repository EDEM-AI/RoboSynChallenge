# Installtion
## Docker (Recommended)
First, please install [**EmbodiChain**](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html), which is the underlying simulation environment we use. We provide the Docker installation method; for other installation methods, please refer to [EmbodiChain Install](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html).
```
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8

mkdir RoboSynChallenge_ws && cd RoboSynChallenge_ws
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
./docker/docker_run.sh <container_name> <data_path>
# This will mount <data_path> to the /root/workspace directory in <container_name> container.
# We recommend setting <data_path> to the RoboSynChallenge_ws/ directory and placing EmbodiChain and RoboSynChallenge there.

```

After enter the docker container, install the `EmbodiChain` package:
```
conda activate py310
cd /root/workspace/EmbodiChain
pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
pip install "numpy<2.0"
```

Then install the `RoboSynChallenge` package:
```
cd /root/worspace
git clone https://github.com/EDEM-AI/RoboSynChallenge.git
cd RoboSynChallenge
pip install -e .
```

Now you can start your data collection!
