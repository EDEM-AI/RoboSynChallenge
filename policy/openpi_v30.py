"""Read LeRobot v3.0 trajectories for the vendored OpenPI training programs."""

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from robosynchallenge.hub import DATASET_REVISION


def open_dataset(repo_id, root, action_keys, action_horizon):
    """Read future actions at the dataset's recorded control frequency."""
    root = Path(root).expanduser().resolve() if root is not None else None
    if root is not None and not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"No LeRobot dataset metadata in {root}.")
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=DATASET_REVISION)
    if meta.info["codebase_version"] != DATASET_REVISION:
        raise ValueError(f"Expected LeRobot {DATASET_REVISION} data at {meta.root}.")
    missing = set(action_keys) - meta.features.keys()
    if missing:
        raise ValueError(
            f"Missing configured action features {sorted(missing)} in {meta.root}."
        )
    return LeRobotDataset(
        repo_id,
        root=root,
        revision=DATASET_REVISION,
        delta_timestamps={
            key: [step / meta.fps for step in range(action_horizon)]
            for key in action_keys
        },
        video_backend="pyav",
        return_uint8=True,
    )


def task_prompt(sample):
    """Use the episode's task sentence returned by the v3.0 reader."""
    return {**sample, "prompt": sample["task"]}
