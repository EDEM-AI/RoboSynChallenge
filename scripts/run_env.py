# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Script to run the environment."""

import argparse
import torch
import numpy as np
import tqdm

import gymnasium as gym
import embodied_challenge

from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args
)
import embodichain.lab.gym.utils.gym_utils as gym_utils
from embodichain.lab.scripts.run_env import generate_and_execute_action_list, preview
from embodichain.utils.logger import log_warning, log_info

gym_utils.DEFAULT_MANAGER_MODULES = gym_utils.DEFAULT_MANAGER_MODULES + [
    "embodied_challenge.managers.actions",
    "embodied_challenge.managers.datasets",
    "embodied_challenge.managers.events",
    "embodied_challenge.managers.observations",
]


def _generate_function(
    env,
    num_traj,
    time_id: int = 0,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    reset_first: bool = True,
    **kwargs,
) -> bool:
    valid = True
    if reset_first:
        _, _ = env.reset()

    while True:
        for trajectory_idx in range(num_traj):
            valid = generate_and_execute_action_list(
                env, trajectory_idx, debug_mode, **kwargs
            )

            if not valid:
                _, _ = env.reset(options={"save_data": False})
                break

        if valid:
            break

        log_warning("Reset valid flag to True.")
        valid = True

    return True


def _get_saved_episode_count(env):
    try:
        dataset_manager = env.get_wrapper_attr("dataset_manager")
    except AttributeError:
        dataset_manager = getattr(
            getattr(env, "unwrapped", env), "dataset_manager", None
        )

    if dataset_manager is None:
        return None

    episode_counts = []
    for mode_cfgs in getattr(dataset_manager, "_mode_functor_cfgs", {}).values():
        for functor_cfg in mode_cfgs:
            functor = getattr(functor_cfg, "func", None)
            if hasattr(functor, "curr_episode"):
                episode_counts.append(int(functor.curr_episode))

    if not episode_counts:
        return None

    return max(episode_counts)


def _generate_until_saved_episode_target(args, env, gym_config, num_traj: int) -> bool:
    target_episodes = int(gym_config.get("max_episodes", 1))
    saved_episodes = _get_saved_episode_count(env)

    if saved_episodes is None:
        return False

    log_info(
        f"Collecting until {target_episodes} successful episodes are saved.",
        color="green",
    )

    _, _ = env.reset()
    saved_episodes = _get_saved_episode_count(env)
    attempt = 0
    progress = tqdm.tqdm(
        total=target_episodes,
        initial=min(saved_episodes, target_episodes),
        desc="Saved successful episodes",
        unit="episode",
    )

    while saved_episodes < target_episodes:
        attempt += 1
        previous_saved_episodes = saved_episodes

        _generate_function(
            env,
            num_traj,
            attempt - 1,
            save_path=getattr(args, "save_path", ""),
            save_video=getattr(args, "save_video", False),
            debug_mode=getattr(args, "debug_mode", False),
            reset_first=False,
            regenerate=getattr(args, "regenerate", False),
        )

        saved_before_reset = _get_saved_episode_count(env)
        _, _ = env.reset(
            options={"save_data": saved_before_reset < target_episodes}
        )
        saved_episodes = _get_saved_episode_count(env)

        progress.update(max(0, min(saved_episodes, target_episodes) - progress.n))

        if saved_episodes == previous_saved_episodes:
            log_warning(
                f"Attempt {attempt} did not save a successful episode "
                f"({saved_episodes}/{target_episodes}). Retrying."
            )
        else:
            log_info(
                f"Saved successful episodes: {saved_episodes}/{target_episodes} "
                f"after {attempt} attempts.",
                color="green",
            )

    progress.close()
    return True


def run_env_main(args, env, gym_config):
    if getattr(args, "preview", False):
        log_info(
            "Preview mode enabled. Launching environment preview...", color="green"
        )
        preview(env)

    log_info("Start offline data generation.", color="green")
    num_traj = 1

    if _generate_until_saved_episode_target(args, env, gym_config, num_traj):
        return

    log_warning(
        "No dataset recorder was found. Falling back to max_episodes generation attempts."
    )
    for i in range(gym_config.get("max_episodes", 1)):
        _generate_function(
            env,
            num_traj,
            i,
            save_path=getattr(args, "save_path", ""),
            save_video=getattr(args, "save_video", False),
            debug_mode=getattr(args, "debug_mode", False),
            regenerate=getattr(args, "regenerate", False),
        )

    _, _ = env.reset()


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()

    add_env_launcher_args_to_parser(parser)

    args = parser.parse_args()

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)

    run_env_main(args, env, gym_config=gym_config)
