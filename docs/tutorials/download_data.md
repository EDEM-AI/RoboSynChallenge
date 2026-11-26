# Download Data

## LeRobot datasets

The default Hub datasets are `han-xudong/cobotmagic_Real_<task>` and
`han-xudong/cobotmagic_Sim_<task>` at revision `v3.0`. Browse the
[Real data](https://huggingface.co/collections/han-xudong/real-data-6b078018b5eb646b7c26fbe1)
and [Sim data](https://huggingface.co/collections/han-xudong/sim-data-6b07801c45756c22865cc334)
collections.

```bash
hf download han-xudong/cobotmagic_Real_click_bell \
  --repo-type dataset --revision v3.0 --local-dir /data/cobotmagic_Real_click_bell
```

Pass a simulation task name, a full Hub dataset ID, or a local v3.0 directory
to `launch/train_policy.sh`. A task name such as `click_bell` resolves to
`han-xudong/cobotmagic_Sim_click_bell`. To use an older local v2.1 copy,
run the importer first.

```bash
uv run --locked --python 3.12 python -m scripts.import_lerobot_v21 \
  /data/raw/cobotmagic_Real_click_bell /data/v30/cobotmagic_Real_click_bell
```

The current simulator writes v3.0 and adds task-program segment annotations.
To regenerate those annotations, collect a new expert episode or record a
physically successful dynamic replay with its source segments. Joint-state
and action training on imported releases can use the original frames after
conversion. The simulation policy expects the CobotMagic action layout.

See [collection and validation](collect_data.md) and
[expert trajectories](task_trajectory.md).

## Checkpoints

The [model checkpoints collection](https://huggingface.co/collections/han-xudong/model-checkpoints-6b07801f9f42e3bccb4a82e0)
contains the ACT, Diffusion and SmolVLA task checkpoints. Omitting the
checkpoint argument to `launch/eval_policy.sh` uses
`han-xudong/ACT_sim_<task>`, `han-xudong/DP_sim_<task>` or
`han-xudong/SmolVLA_sim_<task>` for the selected policy. You can also pass
an explicit Hub ID or local checkpoint directory.

```bash
bash launch/eval_policy.sh act click_bell clear 3 --headless true
hf download han-xudong/ACT_sim_click_bell --local-dir /data/weights/ACT_sim_click_bell
```

- **ACT and Diffusion.** The default Hub releases include the LeRobot
  preprocessor and postprocessor files. For a legacy local checkpoint, the
  evaluation adapter can recover the processor state from embedded statistics.
- **SmolVLA.** The released checkpoint includes the processor files.
  Its backbone and tokenizer use the Hugging Face cache under `HF_HOME`.
- **π₀ and π₀.₅.** OpenPI downloads base weights into `OPENPI_DATA_HOME`.
  Existing OpenPI Orbax directories can be loaded directly. Continue training
  with the original experiment and its optimizer state. Use the normalization
  asset stored with the checkpoint for inference. Compute new data statistics
  when fine-tuning on a different dataset or feature schema. See
  [π₀](policy/pi0.md) and [π₀.₅](policy/pi05.md).

## Simulation assets

Project meshes, URDFs and textures are under `assets/`. Scene and robot
bindings live in `configs/<task>/<setting>/task.cobotmagic.yaml` and
`task_program/`. The launchers default `EMBODICHAIN_DATA_ROOT` to the
repository's `assets/`. EmbodiChain writes download, extraction and robot
assembly caches under that data root. Keep the native bindings and the files
they reference together when copying deployments. The opening-pan task uses
the prepared carrot collision mesh and the single-arm tasks use the released
CobotMagic scene assets.
