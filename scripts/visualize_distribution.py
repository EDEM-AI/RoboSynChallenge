#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
for root in (WORKSPACE / "EmbodiChain", REPO):
    if root.exists() and str(root) not in sys.path: sys.path.insert(0, str(root))

def parse_args():
    parser = argparse.ArgumentParser(description="Overlay object masks from solvable resets.")
    for name in ("env-name", "gym_config", "action_config"): parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--resets", type=int, default=100, help="target solvable sample count")
    parser.add_argument("--max-resets", type=int); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="cam_high"); parser.add_argument("--output-root", default="results/visualize_distribution")
    parser.add_argument("--alpha", type=float, default=0.45); parser.add_argument("--background-mode", choices=("inpaint", "first-frame"), default="inpaint")
    parser.add_argument("--include-first-overlay", action="store_true"); parser.add_argument("--foreground-uids", nargs="*")
    for name, default in (("num_envs", 1), ("arena_space", 5.0), ("gpu_id", 0)): parser.add_argument(f"--{name}", type=type(default), default=default)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--enable_rt", action="store_true")
    parser.add_argument("--headless", dest="headless", action="store_true", default=True); parser.add_argument("--no-headless", dest="headless", action="store_false")
    return parser.parse_args()

def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path

def read_json(path):
    with open(repo_path(path), "r", encoding="utf-8") as handle: return json.load(handle)

def default_foreground(config):
    result = []
    for group in ("rigid_object", "articulation"):
        for obj in config.get(group, []) or []:
            uid, pos = obj.get("uid"), obj.get("init_pos")
            hidden = isinstance(pos, list) and len(pos) > 2 and pos[2] < -5
            if uid and "distractor" not in uid.lower() and not hidden: result.append(uid)
    return result

def make_env(opt):
    import robosynchallenge, gymnasium as gym  # noqa: F401
    import embodichain.lab.gym.utils.gym_utils as gym_utils
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, merge_args_with_gym_config
    from embodichain.lab.sim import SimulationManagerCfg

    for suffix in ("actions", "datasets", "events", "observations"):
        module = f"robosynchallenge.managers.{suffix}"
        if module not in gym_utils.DEFAULT_MANAGER_MODULES:
            gym_utils.DEFAULT_MANAGER_MODULES.append(module)

    raw = read_json(opt.gym_config); config = deepcopy(raw)

    for sensor in config.get("sensor", []) or []:
        if sensor.get("sensor_type") in ("Camera", "StereoCamera"):
            sensor["enable_mask"] = True

    merged = merge_args_with_gym_config(opt, config); env_cfg = config_to_cfg(merged)

    env_cfg.filter_visual_rand = True; env_cfg.filter_dataset_saving = True
    env_cfg.sim_cfg = SimulationManagerCfg(headless=merged["headless"], sim_device=merged["device"], enable_rt=merged["enable_rt"], gpu_id=merged["gpu_id"], arena_space=merged["arena_space"])
    return gym.make(id=merged["id"], cfg=env_cfg, action_config=read_json(opt.action_config)), raw

def as_numpy(value):
    import torch
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value

def camera_frame(obs, camera):
    image = as_numpy(obs["sensor"][camera]["color"])[0][..., :3].astype("uint8"); mask = as_numpy(obs["sensor"][camera]["mask"])[0]
    mask = mask[..., 0] if mask.ndim == 3 else mask
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Camera color/mask size mismatch: image={image.shape}, mask={mask.shape}")
    return image, mask
def save_png(path, image):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(image).save(path)
def main():
    import cv2, numpy as np
    opt = parse_args(); env, config = make_env(opt); base = env.unwrapped if hasattr(env, "unwrapped") else env
    uids = opt.foreground_uids or default_foreground(config)
    user_ids = [int(v) for uid in uids if uid in base.sim.asset_uids for v in np.asarray(as_numpy(base.sim.get_asset(uid).get_user_ids())).reshape(-1)]
    output_dir = repo_path(opt.output_root) / opt.env_name; target = opt.resets; limit = opt.max_resets or opt.resets * 20
    kept, failures, composite = [], [], None
    try:
        for index in range(limit):
            if len(kept) >= target: break
            seed = opt.seed + index
            try:
                if hasattr(base, "affordance_datas"): base.affordance_datas = {}
                obs, _ = env.reset(seed=seed); actions = base.create_demo_action_list()
                if actions is None or len(actions) == 0: raise RuntimeError("create_demo_action_list failed")
                image, mask = camera_frame(obs, opt.camera); object_mask = np.isin(mask, np.asarray(user_ids, dtype=mask.dtype))
                if object_mask.sum() == 0: raise RuntimeError("empty foreground mask")
            except Exception as exc:
                failures.append({"reset_index": index, "seed": seed, "error": str(exc)}); continue
            if composite is None:
                save_png(output_dir / "first_solvable_frame.png", image); save_png(output_dir / "first_solvable_mask.png", object_mask.astype("uint8") * 255)
                mask_u8 = object_mask.astype("uint8") * 255
                composite = image if opt.background_mode == "first-frame" else cv2.inpaint(image, mask_u8, 5, cv2.INPAINT_TELEA)
                save_png(output_dir / "background.png", composite)
                if not opt.include_first_overlay: kept.append({"reset_index": index, "seed": seed, "setup_only": True}); continue
            composite = np.where(object_mask[..., None], (1 - opt.alpha) * composite + opt.alpha * image, composite).astype("uint8")
            kept.append({"reset_index": index, "seed": seed, "mask_pixels": int(object_mask.sum())}); print(f"[{opt.env_name}] kept {len(kept)}/{target}, tried {index + 1}")
        if composite is None: raise RuntimeError("No solvable sample with a non-empty object mask was found.")
        save_png(output_dir / "object_distribution_overlay.png", composite)
        report = {"target": target, "tried": len(kept) + len(failures), "kept": kept, "failures": failures, "foreground_uids": uids}
        with open(output_dir / "report.json", "w", encoding="utf-8") as handle: json.dump(report, handle, indent=2)
        print(f"[{opt.env_name}] wrote {output_dir / 'object_distribution_overlay.png'}")
    finally: env.close()
if __name__ == "__main__": main()
