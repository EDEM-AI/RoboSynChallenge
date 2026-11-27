"""Compatibility launcher for OpenPI normalization on LeRobot v3.0."""

import tyro
from compute_norm_stats import main as compute_norm_stats


def main(config_name: str, base_dir: str | None = None, max_frames: int | None = None):
    compute_norm_stats(config_name, max_frames=max_frames, dataset_root=base_dir)


if __name__ == "__main__":
    tyro.cli(main)
