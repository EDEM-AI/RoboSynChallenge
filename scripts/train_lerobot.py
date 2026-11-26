"""Run the locked LeRobot trainer with a local or published dataset."""

import argparse
import os
from pathlib import Path
import shutil
import sys

from robosynchallenge.hub import DATASET_REVISION, dataset_source


def run_training(policy, dataset, output, options):
    repo_id, root = dataset_source(dataset)
    policy_type = [f"--policy.type={policy}"]
    if any(arg == "--policy.path" or arg.startswith("--policy.path=") for arg in options):
        policy_type = []
    dataset_options = [f"--dataset.repo_id={repo_id}", "--dataset.video_backend=pyav"]
    if root is not None:
        dataset_options.append(f"--dataset.root={root}")
    os.execv(
        sys.executable,
        [sys.executable, "-m", "lerobot.scripts.lerobot_train",
         *policy_type, "--policy.push_to_hub=false", *dataset_options,
         f"--output_dir={output}", *options, f"--dataset.revision={DATASET_REVISION}"],
    )


def main(policy):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    defaults = {
        "batch-size": 8,
        "num-workers": 4,
        "steps": 100000,
        "log-freq": 200,
        "save-freq": 20000,
        "eval-freq": 0,
        "seed": 1000,
    }
    for field, default in defaults.items():
        parser.add_argument(f"--{field}", type=int, default=default)
    parser.add_argument("--job-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-obs-steps", type=int, default=1 if policy == "act" else 2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    if policy == "act":
        parser.add_argument("--chunk-size", type=int, default=16)
    else:
        parser.add_argument("--horizon", type=int, default=16)
        parser.add_argument("--num-inference-steps", type=int)
        parser.add_argument("--crop-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
        parser.add_argument("--img-micro-bs", type=int, nargs="?", const=64, default=0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--repo-id")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="robosynchallenge")
    parser.add_argument("--wandb-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-imagenet-stats", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args, extra = parser.parse_known_args()

    if getattr(args, "img_micro_bs", 0):
        parser.error("Image encoder micro-batching required a custom Diffusion trainer.")
    if args.distributed and "WORLD_SIZE" not in os.environ:
        parser.error("Launch distributed training with accelerate launch or torchrun.")
    if args.local_rank is not None:
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank))

    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not (root / "meta/info.json").is_file():
        parser.error(f"No LeRobot dataset metadata in {root}.")
    training = {
        key.replace("-", "_"): getattr(args, key.replace("-", "_")) for key in defaults
    }
    training["env_eval_freq"] = training.pop("eval_freq")
    training["job_name"] = args.wandb_name or args.job_name
    training["save_checkpoint"] = str(not args.no_save_checkpoint).lower()
    training["wandb.enable"] = str(args.wandb).lower()
    training["wandb.project"] = args.wandb_project

    if args.resume:
        config = output / "checkpoints/last/pretrained_model/train_config.json"
        if not config.is_file():
            parser.error(f"No resume configuration at {config}.")
        training["config_path"] = config
        training["resume"] = "true"
    else:
        if args.overwrite and output.exists():
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                parser.error("Use a fresh output directory for distributed training.")
            if root.is_relative_to(output):
                parser.error(f"The output directory contains the dataset: {output}.")
            shutil.rmtree(output)
        training.update({
            "dataset.video_backend": args.video_backend,
            "dataset.repo_id": args.repo_id,
            "dataset.use_imagenet_stats": str(not args.no_imagenet_stats).lower(),
            "policy.device": args.device,
            "policy.use_amp": str(args.use_amp).lower(),
            "policy.n_obs_steps": args.n_obs_steps,
            "policy.n_action_steps": args.n_action_steps,
        })
        if policy == "act":
            training["policy.chunk_size"] = args.chunk_size
        else:
            training["policy.horizon"] = args.horizon
            training["policy.num_inference_steps"] = args.num_inference_steps
            training["policy.crop_shape"] = args.crop_shape

    overrides = [f"--{key}={value}" for key, value in training.items() if value is not None]
    launcher = Path(__file__).resolve().parents[1] / "launch/train_policy.sh"
    os.execv(
        "/bin/bash",
        ["bash", str(launcher), policy, str(root), str(output), *overrides, *extra],
    )


if __name__ == "__main__":
    run_training(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
