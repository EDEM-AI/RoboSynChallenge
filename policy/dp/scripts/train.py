#!/usr/bin/env python
"""Train LeRobot v0.3.3 DiffusionPolicy on a canonical v2.1 dataset."""

import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

import torch


LEGACY_FEATURE_ALIASES = {
    "observation.state": "observation.qpos",
    "observation.images.cam_high": "cam_high.color",
    "observation.images.cam_right_wrist": "cam_right_wrist.color",
    "observation.images.cam_left_wrist": "cam_left_wrist.color",
}


class _AliasedDatasetMetadata:
    def __init__(self, base_meta, aliases):
        self._base_meta = base_meta
        self._aliases = aliases
        alias_sources = set(aliases.values())

        self._features = {
            key: value
            for key, value in base_meta.features.items()
            if key not in alias_sources and value.get("dtype") not in {"image", "video"}
        }
        for alias_key, source_key in aliases.items():
            self._features[alias_key] = copy.deepcopy(base_meta.features[source_key])

        self._stats = dict(base_meta.stats)
        for alias_key, source_key in aliases.items():
            if source_key in base_meta.stats:
                self._stats[alias_key] = base_meta.stats[source_key]

    def __getattr__(self, name):
        return getattr(self._base_meta, name)

    @property
    def features(self):
        return self._features

    @property
    def stats(self):
        return self._stats

    @property
    def image_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]


class _AliasedLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, aliases):
        self._base_dataset = base_dataset
        self._aliases = aliases
        self.meta = _AliasedDatasetMetadata(base_dataset.meta, aliases)

    def __getattr__(self, name):
        return getattr(self._base_dataset, name)

    def __len__(self):
        return len(self._base_dataset)

    def __getitem__(self, idx):
        item = self._base_dataset[idx]
        for alias_key, source_key in self._aliases.items():
            if source_key not in item:
                continue
            value = item[source_key]
            if alias_key.startswith("observation.images.") and isinstance(value, torch.Tensor):
                if value.ndim == 3:
                    value = value.unsqueeze(0)
                state = item.get("observation.state")
                if state is None:
                    state = item.get("observation.qpos")
                if (
                    isinstance(state, torch.Tensor)
                    and state.ndim >= 2
                    and value.ndim == 4
                    and value.shape[0] == 1
                    and state.shape[0] > 1
                ):
                    value = value.expand(state.shape[0], *value.shape[1:]).clone()
            item[alias_key] = value

            source_pad_key = f"{source_key}_is_pad"
            if source_pad_key in item:
                item[f"{alias_key}_is_pad"] = item[source_pad_key]

        return item


def _legacy_aliases_for_dataset(dataset):
    features = dataset.meta.features
    aliases = {
        alias_key: source_key
        for alias_key, source_key in LEGACY_FEATURE_ALIASES.items()
        if alias_key not in features and source_key in features
    }

    if not aliases:
        return {}
    if "observation.state" not in aliases and "observation.state" not in features:
        return {}
    return aliases


def _patch_lerobot_dataset_factory(distributed_context=None):
    import lerobot.scripts.train as train_module

    original_make_dataset = train_module.make_dataset

    def make_dataset_with_legacy_aliases(cfg):
        dataset = original_make_dataset(cfg)
        aliases = _legacy_aliases_for_dataset(dataset)
        if aliases:
            print("[DP train] Applying legacy dataset feature aliases:")
            for alias_key, source_key in aliases.items():
                print(f"  {source_key} -> {alias_key}")
            dataset = _AliasedLeRobotDataset(dataset, aliases)
        return dataset

    train_module.make_dataset = make_dataset_with_legacy_aliases
    if distributed_context is not None:
        original_make_policy = train_module.make_policy
        rank, world_size, local_rank = distributed_context
        original_dataloader = torch.utils.data.DataLoader
        def make_dataloader(dataset, *args, **kwargs):
            sampler = kwargs.get("sampler")
            if sampler is not None:
                kwargs["sampler"] = list(sampler)[rank::world_size]
            elif kwargs.get("shuffle"):
                kwargs["sampler"] = torch.utils.data.distributed.DistributedSampler(
                    dataset, world_size, rank
                )
                kwargs["shuffle"] = False
            return original_dataloader(dataset, *args, **kwargs)
        torch.utils.data.DataLoader = make_dataloader
        def make_patched_policy(*args, **kwargs):
            policy = original_make_policy(*args, **kwargs)
            ddp_policy = torch.nn.parallel.DistributedDataParallel(
                policy,
                device_ids=[local_rank],
                output_device=local_rank,
            )
            for name in ("config", "get_optim_params", "save_pretrained", "update"):
                if hasattr(policy, name):
                    setattr(ddp_policy, name, getattr(policy, name))
            return ddp_policy
        train_module.make_policy = make_patched_policy
    return train_module.train


def _patch_diffusion_image_encoder(img_micro_bs):
    if not img_micro_bs:
        return
    if img_micro_bs <= 0:
        raise ValueError("--img-micro-bs must be positive.")

    from lerobot.policies.diffusion import modeling_diffusion
    from torch.utils.checkpoint import checkpoint

    original_forward = modeling_diffusion.DiffusionRgbEncoder.forward

    def forward_with_micro_batches(self, images):
        if images.shape[0] <= img_micro_bs:
            return original_forward(self, images)

        outputs = []

        def run_encoder(image_chunk):
            return original_forward(self, image_chunk)

        for image_chunk in images.split(img_micro_bs, dim=0):
            if torch.is_grad_enabled():
                outputs.append(checkpoint(run_encoder, image_chunk, use_reentrant=False))
            else:
                outputs.append(run_encoder(image_chunk))

        return torch.cat(outputs, dim=0)

    modeling_diffusion.DiffusionRgbEncoder.forward = forward_with_micro_batches
    print(f"[DP train] Using image encoder micro-batches: img_micro_bs={img_micro_bs}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--crop-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None,
        help="Image crop size before the DP RGB encoder. Default disables cropping.",
    )
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="robosynchallenge")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-imagenet-stats", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument(
        "--img-micro-bs", type=int, nargs="?", const=64, default=64, metavar="N",
        help="Split and checkpoint DP RGB encoder inputs into micro-batches. Default is 64; set 0 to disable.",
    )
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    distributed_context = None
    if args.distributed:
        import torch.distributed as dist
        local_rank = args.local_rank if args.local_rank is not None else int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        distributed_context = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    repo_id = args.repo_id or dataset_root.name
    output_dir = Path(args.output_dir).expanduser().resolve()
    if distributed_context is not None:
        output_dir = output_dir / f"rank_{distributed_context[0]}"

    if args.overwrite and output_dir.exists() and not args.resume:
        shutil.rmtree(output_dir)

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.default import WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    lerobot_train = _patch_lerobot_dataset_factory(distributed_context)
    _patch_diffusion_image_encoder(args.img_micro_bs)

    policy_kwargs = {
        "device": args.device,
        "use_amp": args.use_amp,
        "push_to_hub": False,
        "n_obs_steps": args.n_obs_steps,
        "horizon": args.horizon,
        "n_action_steps": args.n_action_steps,
        "num_inference_steps": args.num_inference_steps,
        "crop_shape": args.crop_shape,
    }
    policy_kwargs = {
        key: value for key, value in policy_kwargs.items() if value is not None or key == "crop_shape"
    }

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=repo_id,
            root=str(dataset_root),
            use_imagenet_stats=not args.no_imagenet_stats,
            video_backend=args.video_backend,
        ),
        policy=DiffusionConfig(**policy_kwargs),
        output_dir=output_dir,
        job_name=args.wandb_name or args.job_name,
        resume=args.resume,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        save_checkpoint=not args.no_save_checkpoint,
        save_freq=args.save_freq,
        wandb=WandBConfig(enable=args.wandb and (distributed_context is None or distributed_context[0] == 0), project=args.wandb_project),
    )
    if args.resume:
        config_path = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Resume config not found: {config_path}")
        sys.argv.append(f"--config_path={config_path}")
        cfg = TrainPipelineConfig.from_pretrained(config_path)
        cfg.output_dir = output_dir
        cfg.job_name = args.wandb_name or args.job_name
        cfg.resume = True
        cfg.num_workers = args.num_workers
        cfg.batch_size = args.batch_size
        cfg.steps = args.steps
        cfg.eval_freq = args.eval_freq
        cfg.log_freq = args.log_freq
        cfg.save_checkpoint = not args.no_save_checkpoint
        cfg.save_freq = args.save_freq
        cfg.wandb = WandBConfig(
            enable=args.wandb and (distributed_context is None or distributed_context[0] == 0),
            project=args.wandb_project,
        )
    lerobot_train(cfg)
    if distributed_context is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
