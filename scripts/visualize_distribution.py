"""Overlay the initial object positions of physically successful native episodes."""

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from PIL import Image

from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.utils import gym_utils
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim import SimulationManager

from robosynchallenge.deployments import deployment_path
from scripts.eval_policy import prepare_episode_reset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("setting", choices=("clear", "random"))
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--max-resets", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="cam_high")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/visualize_distribution")
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--background-mode", choices=("inpaint", "first-frame"), default="inpaint"
    )
    parser.add_argument("--include-first-overlay", action="store_true")
    parser.add_argument("--foreground-uids", nargs="+")
    parser.add_argument("--affordance-branch", type=int)
    parser.add_argument("--renderer", default="fast-rt")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    args.max_resets = (
        args.max_resets if args.max_resets is not None else args.resets * 20
    )
    if args.resets < 1 or args.max_resets < args.resets or not 0 <= args.alpha <= 1:
        parser.error("Require 1 <= resets <= max-resets and 0 <= alpha <= 1.")
    if args.seed < 0 or (
        args.affordance_branch is not None and args.affordance_branch < 0
    ):
        parser.error("Seed and affordance branch must be non-negative.")
    return args


def make_env(opts):
    discover_task_packages()
    execute_init_hooks()
    parser = argparse.ArgumentParser()
    gym_utils.add_env_launcher_args_to_parser(parser, require_gym_config=True)
    args = parser.parse_args(
        ["--gym_config", str(deployment_path(opts.task, opts.setting))]
    )
    args.headless = opts.headless
    args.renderer = opts.renderer
    args.gpu_id = opts.gpu_id
    args.num_envs = 1
    args.filter_dataset_saving = True
    args.filter_visual_rand = True

    cfg, config, _ = gym_utils.build_env_cfg_from_args(args)
    for camera in cfg.sensor:
        if camera.uid == opts.camera:
            camera.enable_mask = True
            break
    else:
        raise ValueError(f"No camera named {opts.camera!r} in the deployment.")
    return gym.make(config["id"], cfg=cfg), config


def scene_foreground(config):
    return [
        asset["uid"]
        for group in ("rigid_object", "articulation")
        for asset in config.get(group, ())
        if not asset["uid"].startswith("distractor")
        and asset.get("init_pos", [0, 0, 0])[2] >= -5
    ]


def save_report(env, config, opts):
    from scripts.collect_seeded import NativeAffordanceBranch

    names = opts.foreground_uids or scene_foreground(config)
    user_ids = [
        value
        for name in names
        for value in env.unwrapped.sim.get_asset(name)
        .get_user_ids()
        .reshape(-1)
        .tolist()
    ]
    if not user_ids:
        raise ValueError("No foreground assets were selected.")
    folder = opts.output_root / opts.task / opts.setting
    folder.mkdir(parents=True, exist_ok=True)
    accepted, failures = [], []
    composite = None
    for seed in range(opts.seed, opts.seed + opts.max_resets):
        prepare_episode_reset(env, seed)
        obs, _ = env.reset(seed=seed, options={"save_data": False})
        frame = obs["sensor"][opts.camera]
        image = frame["color"][0, ..., :3].cpu().numpy().astype(np.uint8, copy=True)
        mask = np.isin(
            frame["mask"][0].cpu().numpy().reshape(image.shape[:2]), user_ids
        )
        if not mask.any():
            failures.append({"seed": seed, "reason": "empty foreground mask"})
            continue
        plan_options = {}
        if opts.affordance_branch is not None:
            plan_options["plan_transform_factory"] = NativeAffordanceBranch(
                seed, opts.affordance_branch
            )
        result = execute_demo_episode(env, **plan_options)
        if not all(result.success):
            failures.append({"seed": seed, "reason": "expert execution failed"})
            continue

        mask_image = mask.astype(np.uint8) * 255
        if composite is None:
            Image.fromarray(image).save(folder / "first_solvable_frame.png")
            Image.fromarray(mask_image).save(folder / "first_solvable_mask.png")
            composite = image.copy()
            if opts.background_mode == "inpaint":
                composite = cv2.inpaint(image, mask_image, 5, cv2.INPAINT_TELEA)
            Image.fromarray(composite).save(folder / "background.png")
            if opts.include_first_overlay:
                composite[mask] = (1 - opts.alpha) * composite[
                    mask
                ] + opts.alpha * image[mask]
        else:
            composite[mask] = (1 - opts.alpha) * composite[mask] + opts.alpha * image[
                mask
            ]

        accepted.append({"seed": seed, "mask_pixels": int(mask.sum())})
        print(
            f"{opts.task}/{opts.setting} accepted {len(accepted)}/{opts.resets} resets",
            flush=True,
        )
        if len(accepted) == opts.resets:
            break

    report = {
        "target": opts.resets,
        "kept": accepted,
        "failures": failures,
        "foreground_uids": names,
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if composite is not None:
        Image.fromarray(composite).save(folder / "object_distribution_overlay.png")
    if len(accepted) != opts.resets:
        raise RuntimeError(
            f"Accepted {len(accepted)}/{opts.resets} resets. See {folder / 'report.json'}."
        )


def main():
    args = parse_args()
    env, config = make_env(args)
    try:
        save_report(env, config, args)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
