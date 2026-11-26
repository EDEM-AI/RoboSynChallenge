"""Compatibility entry point for the native EmbodiChain task launcher."""

import argparse
import os
import sys

from embodichain.cli.main import main as embodichain_cli

from robosynchallenge.deployments import ROOT, deployment_from_path


def replay_legacy(arguments):
    if "--replay" not in arguments:
        return False
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--replay_trajectory")
    parser.add_argument("--replay_mode", choices=("kinematic", "dynamic", "control"), default="kinematic")
    replay, _ = parser.parse_known_args(arguments)
    if not replay.replay_trajectory:
        return False

    import torch

    trajectory = torch.load(replay.replay_trajectory, map_location="cpu", weights_only=False)
    if not isinstance(trajectory, dict) or not {"state", "action"}.issubset(trajectory):
        return False

    import gymnasium as gym
    from embodichain.lab.gym.utils import gym_utils
    from embodichain.lab.gym.utils.registration import discover_task_packages, execute_init_hooks
    from embodichain.lab.sim import SimulationManager

    from robosynchallenge.replay import replay_trajectory

    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    gym_utils.add_env_launcher_args_to_parser(parser, require_gym_config=True)
    args, _ = parser.parse_known_args(arguments)
    args.filter_dataset_saving = True
    discover_task_packages()
    execute_init_hooks()
    cfg, scene, action_cfg = gym_utils.build_env_cfg_from_args(args)
    env = gym.make(scene["id"], cfg=cfg, **action_cfg)
    try:
        replay_trajectory(env, replay.replay_trajectory, mode=replay.replay_mode)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
    return True


def main():
    os.environ.setdefault("EMBODICHAIN_DATA_ROOT", str(ROOT / "assets"))
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
        os.environ.setdefault("SDL_VIDEO_DRIVER", "offscreen")
    args = list(sys.argv[1:])
    for index, arg in enumerate(args):
        if arg in {"--gym_config", "--gym-config"} and index + 1 < len(args):
            path = deployment_from_path(args[index + 1])
            args[index + 1] = str(path)
        elif arg.startswith(("--gym_config=", "--gym-config=")):
            flag, value = arg.split("=", 1)
            path = deployment_from_path(value)
            args[index] = f"{flag}={path}"
        else:
            continue

        if path.suffix in {".yml", ".yaml"}:
            for action_index, action_arg in enumerate(args):
                if action_arg == "--action_config":
                    del args[action_index : action_index + 2]
                    break
                if action_arg.startswith("--action_config="):
                    del args[action_index]
                    break
        break
    if not replay_legacy(args):
        embodichain_cli(["run-task", *args])


if __name__ == "__main__":
    main()
