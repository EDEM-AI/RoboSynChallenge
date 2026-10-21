#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--lerobot-root", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_runtime(args: argparse.Namespace):
    if args.lerobot_root:
        lerobot_src = Path(args.lerobot_root).expanduser().resolve() / "src"
        if lerobot_src.exists():
            sys.path.insert(0, str(lerobot_src))
        else:
            raise FileNotFoundError(
                f"LeRobot source path does not exist: {lerobot_src}. "
                "Install lerobot in this Python environment or set SMOLVLA_LEROBOT_ROOT."
            )

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import lerobot. Install LeRobot in the SmolVLA Python env, "
            "or run policy/smolvla/setup_lerobot.sh and set SMOLVLA_LEROBOT_ROOT."
        ) from exc

    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    policy.to(args.device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def _load_obs(obs_path: str) -> tuple[dict[str, np.ndarray], str]:
    with np.load(obs_path, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files if key != "task"}
        task = str(data["task"].item()) if "task" in data else ""
    return obs, task


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _clock(device: str) -> float:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _infer(policy, preprocessor, postprocessor, request: dict, device: str) -> dict:
    # Disk transport is not model inference; start after loading the request.
    obs, task = _load_obs(request["obs_path"])
    started_at = _clock(device)
    obs = {key: torch.from_numpy(value) for key, value in obs.items()}
    obs["task"] = task
    proc_obs = preprocessor(obs)
    preparation_s = _clock(device) - started_at

    actions, durations_s, inference_action_indices = [], [], []
    for index in range(int(request.get("n_action_steps", 1))):
        action_queue = getattr(policy, "_action_queue", None)
        if action_queue is None:
            action_queue = getattr(policy, "_queues", {}).get("action")
        runs_model_inference = action_queue is None or len(action_queue) == 0
        started_at = _clock(device) if runs_model_inference else None
        action = policy.select_action(proc_obs)
        action = postprocessor(action)
        actions.append(action.squeeze(0).cpu().numpy().astype(np.float32).tolist())
        if runs_model_inference:
            elapsed = _clock(device) - started_at
            # Preprocessing is shared by this request, not repeated per cached action.
            durations_s.append(elapsed + (preparation_s if not durations_s else 0.0))
            inference_action_indices.append(index)

    return {"ok": True, "actions": actions, "durations_s": durations_s,
            "inference_action_indices": inference_action_indices}


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy, preprocessor, postprocessor = _load_runtime(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        cmd = request.get("cmd")

        try:
            if cmd == "reset":
                policy.reset()
                _send({"ok": True})
                continue

            if cmd != "infer":
                raise ValueError(f"Unsupported command: {cmd}")

            with contextlib.redirect_stdout(sys.stderr):
                response = _infer(policy, preprocessor, postprocessor, request, args.device)

            _send(response)
        except Exception as exc:  # noqa: BLE001
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
