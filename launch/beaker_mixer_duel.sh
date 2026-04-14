#!/bin/bash
python -m scripts.run_env \
    --gym_config configs/beaker_mixer/gym_config_duel.json \
    --action_config configs/beaker_mixer/action_config_duel.json \
    --num_envs 1 \
    --enable_rt
