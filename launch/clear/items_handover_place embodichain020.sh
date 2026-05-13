#!/bin/bash
python -m scripts.run_env \
    --gym_config configs/items_handover_place/gym_config_clear.json \
    --action_config configs/items_handover_place/action_config.json \
    --num_envs 1 \
    --renderer  fast-rt \

    # --filter_visual_rand \
