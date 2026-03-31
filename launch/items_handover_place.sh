#!/bin/bash
python -m scripts.run_env \
    --gym_config configs/items_handover_place/gym_config.json \
    --action_config configs/items_handover_place/action_config.json \
    --filter_visual_rand \
    --num_envs 1 \