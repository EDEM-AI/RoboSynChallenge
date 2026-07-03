# Download Data

Pre-collected RoboSynChallenge datasets are hosted on Hugging Face. Store each downloaded dataset under the standard local LeRobot id `RoboSynChallenge/cobotmagic_Real_{task_name}` or `RoboSynChallenge/cobotmagic_Sim_{task_name}`.

The dataset task names are: `click_bell`, `handle_basket`, `water_pouring`, `table_rearrangement`, `items_handover`, `drawer_open_place`, `mixer_operating`, `item_assembly`, `manipulate_pipette`, `sample_loading`, and `open_pan`.

## Real Data

For example, download the Hugging Face source dataset `RoboSynChallenge/cobotmagic_Real_click_bell` into the `click_bell` real-data id:

```bash
cd /root/workspace/RoboSynChallenge
huggingface-cli download RoboSynChallenge/cobotmagic_Real_click_bell \
    --repo-type dataset \
    --local-dir lerobot_dataset/RoboSynChallenge/cobotmagic_Real_click_bell
```

With `HF_LEROBOT_HOME=/root/workspace/RoboSynChallenge/lerobot_dataset`, this local dataset is addressed by `repo_id="RoboSynChallenge/cobotmagic_Real_click_bell"`.

## Sim Data

For the same local layout, generated or downloaded simulation data can be stored as:

```text
lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_{task_name}
```

With the same `HF_LEROBOT_HOME`, its `repo_id` is `RoboSynChallenge/cobotmagic_Sim_{task_name}`. For `click_bell`, use `RoboSynChallenge/cobotmagic_Sim_click_bell`.

## Training

For fine-tuning with downloaded datasets, see <a href="policy/pi0.html#train-with-downloaded-data">PI0 training with downloaded data</a> and <a href="policy/pi05.html#train-with-downloaded-data">PI0.5 training with downloaded data</a>.
