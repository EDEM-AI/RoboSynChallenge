"""Backfill left/right end-effector poses into an existing LeRobot dataset.

The old EmbodiChain LeRobot recorder only saved hard-coded robot fields
(`observation.qpos`, `observation.qvel`, `observation.qf`). If end-effector
poses were configured under `robot/...`, they were present in the rollout buffer
but not written into the final LeRobot parquet files. This script reconstructs
those poses from saved qpos using the same EmbodiChain FK path as
`get_robot_eef_pose`:

    qpos[:, robot.get_joint_ids(part_name)] -> robot.compute_fk(..., to_matrix=True)

By default the script copies the dataset to a new directory and modifies the
copy. Pass `--in-place` to update the original dataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
EMBODICHAIN_ROOT = WORKSPACE_ROOT / "EmbodiChain"
CHALLENGE_ROOT = WORKSPACE_ROOT / "Embodied_Challenge"

for path in (WORKSPACE_ROOT, EMBODICHAIN_ROOT, CHALLENGE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


POSE_FIELDS = {
    "right_ee_pose": "right_arm",
    "left_ee_pose": "left_arm",
}



# python3 Embodied_Challenge/scripts/add_lerobot_eef_pose.py \
#   --dataset Embodied_Challenge/lerobot_dataset/cobotmagic_Sim_manipulate_mixer_dual_046 \
#   --gym_config Embodied_Challenge/configs/manipulate_pipette/gym_config.json \
#   --device cuda

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use EmbodiChain FK to add right_ee_pose and left_ee_pose to an "
            "existing LeRobot dataset."
        )
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset",
        type=Path,
        help="Path to the LeRobot dataset directory, e.g. .../cobotmagic_xxx.",
    )
    dataset_group.add_argument(
        "--repo-id",
        help=(
            "Dataset repo id/name under --root, e.g. "
            "cobotmagic_Sim_manipulate_mixer_dual."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=CHALLENGE_ROOT / "lerobot_dataset",
        help="LeRobot dataset root used together with --repo-id.",
    )
    parser.add_argument(
        "--gym_config",
        type=Path,
        required=True,
        help="Gym config matching the robot used to collect the dataset.",
    )
    parser.add_argument(
        "--action_config",
        type=Path,
        default=None,
        help="Optional action config. Usually not needed for FK backfill.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help=(
            "Output dataset directory. If omitted and not --in-place, the "
            "script writes to '<dataset>_with_eef_pose'."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the input dataset directory directly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing right_ee_pose / left_ee_pose columns.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Remove --output-dataset first if it already exists.",
    )
    parser.add_argument(
        "--qpos-key",
        default="observation.qpos",
        help="Qpos column in the LeRobot parquet files.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used to build the EmbodiChain env and run FK.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id forwarded to EmbodiChain SimulationManagerCfg.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Number of frames per FK batch.",
    )
    parser.add_argument(
        "--keep-sensors",
        action="store_true",
        help="Keep sensors from gym_config when building the FK env.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Do not update meta/stats.json for the two new fields.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build env, inspect mappings, but do not write parquet/meta files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset is not None:
        return args.dataset.expanduser().resolve()
    return (args.root.expanduser() / args.repo_id).resolve()


def prepare_output_dataset(
    source_dataset: Path,
    output_dataset: Path | None,
    in_place: bool,
    overwrite_output: bool,
    dry_run: bool,
) -> Path:
    if in_place:
        return source_dataset

    target = output_dataset
    if target is None:
        target = source_dataset.with_name(f"{source_dataset.name}_with_eef_pose")
    target = target.expanduser().resolve()

    if dry_run:
        logging.info("Dry run: would copy dataset %s -> %s", source_dataset, target)
        return source_dataset

    if target.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"Output dataset already exists: {target}. "
                "Use --overwrite-output or choose another --output-dataset."
            )
        shutil.rmtree(target)

    logging.info("Copying dataset %s -> %s", source_dataset, target)
    shutil.copytree(source_dataset, target)
    return target


def build_fk_env(gym_config_path: Path, action_config_path: Path | None, args: argparse.Namespace):
    """Build a minimal EmbodiChain env so robot.compute_fk matches collection."""

    import gymnasium as gym
    import embodied_challenge  # noqa: F401 - registers challenge envs
    import embodichain.lab.gym.utils.gym_utils as gym_utils
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.sim import SimulationManagerCfg

    challenge_modules = [
        "embodied_challenge.managers.actions",
        "embodied_challenge.managers.datasets",
        "embodied_challenge.managers.events",
        "embodied_challenge.managers.observations",
    ]
    for module in challenge_modules:
        if module not in gym_utils.DEFAULT_MANAGER_MODULES:
            gym_utils.DEFAULT_MANAGER_MODULES.append(module)

    gym_config = load_json(gym_config_path)
    gym_config = copy.deepcopy(gym_config)

    gym_config["num_envs"] = 1
    gym_config["device"] = args.device
    gym_config["headless"] = True
    gym_config["enable_rt"] = False
    gym_config["gpu_id"] = args.gpu_id
    gym_config.setdefault("arena_space", 4.0)

    if not args.keep_sensors:
        gym_config["sensor"] = []

    env_cfg = config_to_cfg(gym_config, manager_modules=challenge_modules)
    env_cfg.filter_dataset_saving = True
    env_cfg.filter_visual_rand = True
    env_cfg.init_rollout_buffer = False
    env_cfg.sim_cfg = SimulationManagerCfg(
        headless=True,
        sim_device=gym_config["device"],
        enable_rt=False,
        gpu_id=gym_config["gpu_id"],
        arena_space=gym_config["arena_space"],
    )

    action_kwargs: dict[str, Any] = {}
    if action_config_path is not None:
        action_config = load_json(action_config_path)
        action_kwargs["action_config"] = action_config

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_kwargs)
    return env


def unwrap_env(env):
    """Return the underlying EmbodiChain env hidden by Gymnasium wrappers."""

    return env.unwrapped if hasattr(env, "unwrapped") else env


def get_qpos_feature_names(info: dict[str, Any], qpos_key: str) -> list[str]:
    try:
        names = info["features"][qpos_key]["names"]
    except KeyError as exc:
        raise KeyError(f"Cannot find feature names for qpos key '{qpos_key}'.") from exc

    if not names:
        raise ValueError(
            f"Feature '{qpos_key}' does not define joint names. "
            "Cannot safely map qpos columns to robot FK joints."
        )
    return list(names)


def build_part_column_indices(
    env,
    dataset_joint_names: list[str],
    part_name: str,
) -> list[int]:
    robot = env.robot
    joint_ids = robot.get_joint_ids(name=part_name, remove_mimic=True)
    part_joint_names = [robot.joint_names[joint_id] for joint_id in joint_ids]

    name_to_col = {name: idx for idx, name in enumerate(dataset_joint_names)}
    missing = [name for name in part_joint_names if name not in name_to_col]
    if missing:
        raise ValueError(
            f"Dataset qpos is missing joints required by '{part_name}': {missing}. "
            f"Dataset joints: {dataset_joint_names}"
        )
    return [name_to_col[name] for name in part_joint_names]


def compute_fk_batched(
    env,
    qpos: np.ndarray,
    part_name: str,
    batch_size: int,
) -> np.ndarray:
    poses: list[np.ndarray] = []
    robot = env.robot
    device = env.device

    for start in range(0, len(qpos), batch_size):
        qpos_batch_np = qpos[start : start + batch_size]
        qpos_batch = torch.as_tensor(qpos_batch_np, dtype=torch.float32, device=device)
        env_ids = torch.zeros(qpos_batch.shape[0], dtype=torch.long, device=device)

        try:
            pose_batch = robot.compute_fk(
                name=part_name,
                qpos=qpos_batch,
                env_ids=env_ids,
                to_matrix=True,
            )
        except Exception:
            if qpos_batch.shape[0] == 1:
                raise
            logging.warning(
                "Batched FK failed for %s with batch=%d; falling back to per-frame FK.",
                part_name,
                qpos_batch.shape[0],
            )
            single_poses = []
            for row in qpos_batch:
                single_pose = robot.compute_fk(
                    name=part_name,
                    qpos=row.unsqueeze(0),
                    env_ids=torch.zeros(1, dtype=torch.long, device=device),
                    to_matrix=True,
                )
                single_poses.append(single_pose)
            pose_batch = torch.cat(single_poses, dim=0)

        poses.append(pose_batch.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(poses, axis=0)


def pose_array_to_arrow(poses: np.ndarray) -> pa.Array:
    pose_type = pa.list_(pa.list_(pa.float32(), list_size=4), list_size=4)
    return pa.array(poses.tolist(), type=pose_type)


def replace_or_append_column(table: pa.Table, name: str, values: pa.Array, overwrite: bool) -> pa.Table:
    if name in table.column_names:
        if not overwrite:
            raise ValueError(
                f"Column '{name}' already exists. Use --overwrite to recompute it."
            )
        return table.set_column(table.column_names.index(name), name, values)
    return table.append_column(name, values)


def table_qpos_to_numpy(table: pa.Table, qpos_key: str) -> np.ndarray:
    if qpos_key not in table.column_names:
        raise KeyError(f"Parquet file does not contain qpos column '{qpos_key}'.")
    return np.asarray(table[qpos_key].to_pylist(), dtype=np.float32)


def update_info_features(info_path: Path, overwrite: bool, dry_run: bool) -> None:
    info = load_json(info_path)
    features = info.setdefault("features", {})

    for field in POSE_FIELDS:
        if field in features and not overwrite:
            raise ValueError(
                f"Feature '{field}' already exists in {info_path}. Use --overwrite."
            )
        features[field] = {
            "dtype": "float32",
            "shape": [4, 4],
            "names": ["row", "column"],
        }

    if dry_run:
        logging.info("Dry run: would update %s features with %s", info_path, list(POSE_FIELDS))
        return

    write_json(info_path, info)


def compute_stats(values: np.ndarray) -> dict[str, Any]:
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    quantiles = np.quantile(flat, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)

    def matrix_list(array: np.ndarray) -> list[list[float]]:
        return array.reshape(4, 4).astype(float).tolist()

    return {
        "min": matrix_list(np.min(flat, axis=0)),
        "max": matrix_list(np.max(flat, axis=0)),
        "mean": matrix_list(np.mean(flat, axis=0)),
        "std": matrix_list(np.std(flat, axis=0)),
        "count": [int(values.shape[0])],
        "q01": matrix_list(quantiles[0]),
        "q10": matrix_list(quantiles[1]),
        "q50": matrix_list(quantiles[2]),
        "q90": matrix_list(quantiles[3]),
        "q99": matrix_list(quantiles[4]),
    }


def update_stats_json(stats_path: Path, all_poses: dict[str, list[np.ndarray]], dry_run: bool) -> None:
    stats = load_json(stats_path) if stats_path.exists() else {}
    for field, chunks in all_poses.items():
        if not chunks:
            continue
        values = np.concatenate(chunks, axis=0)
        stats[field] = compute_stats(values)

    if dry_run:
        logging.info("Dry run: would update %s stats for %s", stats_path, list(all_poses))
        return

    write_json(stats_path, stats)


def add_pose_columns_to_dataset(
    dataset_path: Path,
    env,
    qpos_key: str,
    overwrite: bool,
    batch_size: int,
    skip_stats: bool,
    dry_run: bool,
) -> None:
    info_path = dataset_path / "meta" / "info.json"
    stats_path = dataset_path / "meta" / "stats.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")

    info = load_json(info_path)
    dataset_joint_names = get_qpos_feature_names(info, qpos_key)
    part_columns = {
        field: build_part_column_indices(env, dataset_joint_names, part_name)
        for field, part_name in POSE_FIELDS.items()
    }

    logging.info("Dataset qpos joints: %s", dataset_joint_names)
    for field, indices in part_columns.items():
        names = [dataset_joint_names[i] for i in indices]
        logging.info("%s <- %s columns %s", field, POSE_FIELDS[field], names)

    data_files = sorted((dataset_path / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")

    if dry_run:
        logging.info("Dry run: found %d parquet files; no parquet will be written.", len(data_files))
        update_info_features(info_path, overwrite=overwrite, dry_run=True)
        return

    all_poses: dict[str, list[np.ndarray]] = {field: [] for field in POSE_FIELDS}

    for parquet_path in tqdm.tqdm(data_files, desc="Backfilling eef pose parquet"):
        table = pq.read_table(parquet_path)
        qpos = table_qpos_to_numpy(table, qpos_key)

        for field, part_name in POSE_FIELDS.items():
            qpos_part = qpos[:, part_columns[field]]
            poses = compute_fk_batched(
                env=env,
                qpos=qpos_part,
                part_name=part_name,
                batch_size=batch_size,
            )
            table = replace_or_append_column(
                table=table,
                name=field,
                values=pose_array_to_arrow(poses),
                overwrite=overwrite,
            )
            if not skip_stats:
                all_poses[field].append(poses)

        pq.write_table(table, parquet_path)

    update_info_features(info_path, overwrite=overwrite, dry_run=False)
    if not skip_stats:
        update_stats_json(stats_path, all_poses, dry_run=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    source_dataset = resolve_dataset_path(args)
    if not source_dataset.exists():
        raise FileNotFoundError(f"Dataset does not exist: {source_dataset}")

    output_dataset = prepare_output_dataset(
        source_dataset=source_dataset,
        output_dataset=args.output_dataset,
        in_place=args.in_place,
        overwrite_output=args.overwrite_output,
        dry_run=args.dry_run,
    )

    env = None
    try:
        env = build_fk_env(args.gym_config, args.action_config, args)
        fk_env = unwrap_env(env)
        logging.info("Built FK env with robot joints: %s", fk_env.robot.joint_names)
        add_pose_columns_to_dataset(
            dataset_path=output_dataset,
            env=fk_env,
            qpos_key=args.qpos_key,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
            skip_stats=args.skip_stats,
            dry_run=args.dry_run,
        )
    finally:
        if env is not None:
            env.close()

    logging.info("Done. Dataset with eef poses: %s", output_dataset)


if __name__ == "__main__":
    main()
