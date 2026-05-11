#!/bin/bash
python -m scripts.run_env \
    --gym_config configs/beaker_mixer/gym_config_dual_clear.json \
    --action_config configs/beaker_mixer/action_config_dual.json \
    --num_envs 1 \
    --enable_rt \
    --headless \
    # --filter_visual_rand \

