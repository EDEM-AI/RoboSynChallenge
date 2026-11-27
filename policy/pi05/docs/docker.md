# Docker serving

Build or run the JAX policy container from the repository root. The image
reads the same Python 3.12 dependency lock as the local launcher. Install
Docker with GPU support before starting the server. See the
[serving guide](../README.md#serve-a-checkpoint) for the `docker run` command.

The Compose entry point also accepts the original `SERVER_ARGS` environment
variable. Set `OPENPI_DATA_HOME` to your weights and tokenizer cache. Set
`OPENPI_CHECKPOINT_DIR` to the local checkpoint tree when serving a fine-tuned
checkpoint. Checkpoint paths starting with `checkpoints/` are relative to the
mounted tree inside the container.

```bash
export OPENPI_DATA_HOME=/data/openpi-cache
export OPENPI_CHECKPOINT_DIR=/data/openpi-checkpoints
export SERVER_ARGS='policy:checkpoint --policy.config pi05_base_robosynchallenge_full --policy.dir checkpoints/my-run/20000 --policy.asset-id cobotmagic'
docker compose -f policy/pi05/scripts/docker/compose.yml up --build
```
