"""Compute normalization statistics for EmbodiChain LeRobot datasets quickly.

This script is a fast path for datasets produced by RoboSynChallenge /
EmbodiChain. It avoids video decoding and the generic HuggingFace parquet scan,
but keeps the normalization *semantics* aligned with ``compute_norm_stats.py``:

- read only episodes declared by ``meta/episodes.jsonl``;
- map ``observation.qpos`` -> ``state`` and ``action`` -> ``actions``;
- build the same action chunks as LeRobot ``delta_timestamps``;
- apply the config's ``DeltaActions`` transform when present;
- update ``RunningStats`` with the same batch size/drop-last behavior.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as transforms

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
except Exception:  # pragma: no cover - only used when LeRobot is unavailable.
    HF_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_dataset_dir(base_dir: str | None, repo_id: str | None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser()
    if repo_id is None:
        raise ValueError("Either base_dir must be provided or config must have repo_id.")
    return Path(HF_LEROBOT_HOME) / repo_id


def _episode_file_path(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    rel_path = data_path.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index)
    return dataset_dir / rel_path


def _detect_columns(info: dict[str, Any], state_col: str | None, action_col: str | None) -> tuple[str, str]:
    features = info.get("features", {})

    if state_col is None:
        for candidate in ("observation.qpos", "observation.state"):
            if candidate in features:
                state_col = candidate
                break
    if action_col is None:
        for candidate in ("action", "actions"):
            if candidate in features:
                action_col = candidate
                break

    if state_col is None or action_col is None:
        raise ValueError(
            "Could not infer state/action parquet columns. "
            f"Available feature keys: {sorted(features.keys())}"
        )
    if state_col not in features:
        raise ValueError(f"State column '{state_col}' is not present in meta/info.json features.")
    if action_col not in features:
        raise ValueError(f"Action column '{action_col}' is not present in meta/info.json features.")

    return state_col, action_col


def _get_delta_action_mask(data_config: _config.DataConfig) -> np.ndarray | None:
    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, transforms.DeltaActions):
            if transform.mask is None:
                return None
            return np.asarray(transform.mask, dtype=bool)
    return None


def _stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    values = df[column].to_numpy()
    if len(values) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(np.stack(values), dtype=np.float32)


def _load_episode_arrays(path: Path, state_col: str, action_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path, columns=[state_col, action_col])
    states = _stack_column(df, state_col)
    actions = _stack_column(df, action_col)
    if len(states) != len(actions):
        raise ValueError(f"State/action row count mismatch in {path}: {len(states)} != {len(actions)}")
    return states, actions


def _build_action_chunks(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    frame_indices = np.arange(len(actions))[:, None]
    offsets = np.arange(action_horizon)[None, :]
    query_indices = np.minimum(frame_indices + offsets, len(actions) - 1)
    return actions[query_indices]


def _apply_delta_actions(states: np.ndarray, action_chunks: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return action_chunks

    dims = min(mask.shape[-1], states.shape[-1], action_chunks.shape[-1])
    if dims == 0:
        return action_chunks

    transformed = action_chunks.copy()
    state_delta = np.where(mask[:dims], states[:, :dims], 0.0)
    transformed[..., :dims] -= state_delta[:, None, :]
    return transformed


def _warn_about_extra_parquet(dataset_dir: Path, expected_files: list[Path]) -> None:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        return

    actual = {path.resolve() for path in data_dir.glob("**/*.parquet")}
    expected = {path.resolve() for path in expected_files}
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)

    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} episode parquet files declared by metadata:\n{preview}")

    if extra:
        preview = "\n".join(str(path) for path in extra[:10])
        print(
            "\nWarning: found parquet files under data/ that are not declared by meta/episodes.jsonl. "
            "They will be ignored by this fast script because LeRobot would otherwise read them via data_dir.\n"
            f"Extra parquet count: {len(extra)}\n{preview}\n"
        )


def _update_stats_in_batches(
    stats: dict[str, normalize.RunningStats],
    state_parts: list[np.ndarray],
    action_parts: list[np.ndarray],
    batch_size: int,
) -> None:
    states = np.concatenate(state_parts, axis=0)
    actions = np.concatenate(action_parts, axis=0)
    stats["state"].update(states)
    stats["actions"].update(actions)


class _EpisodeCache:
    def __init__(self, max_size: int, state_col: str, action_col: str):
        self._max_size = max_size
        self._state_col = state_col
        self._action_col = action_col
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, episode_index: int, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if episode_index in self._cache:
            self._cache.move_to_end(episode_index)
            return self._cache[episode_index]

        arrays = _load_episode_arrays(path, self._state_col, self._action_col)
        self._cache[episode_index] = arrays
        self._cache.move_to_end(episode_index)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return arrays


def _process_sequential(
    *,
    stats: dict[str, normalize.RunningStats],
    episodes: list[dict[str, Any]],
    episode_files: list[Path],
    state_col: str,
    action_col: str,
    action_horizon: int,
    delta_action_mask: np.ndarray | None,
    batch_size: int,
    max_samples: int,
) -> tuple[int, int]:
    buffered_states: list[np.ndarray] = []
    buffered_actions: list[np.ndarray] = []
    buffered_count = 0
    processed_samples = 0
    processed_files = 0

    for episode, episode_file in tqdm(list(zip(episodes, episode_files, strict=True)), desc="Processing episodes"):
        if processed_samples >= max_samples:
            break

        states, actions = _load_episode_arrays(episode_file, state_col, action_col)
        expected_length = int(episode["length"])
        if len(states) != expected_length:
            raise ValueError(
                f"Episode {episode['episode_index']} length mismatch: "
                f"metadata says {expected_length}, parquet has {len(states)} rows"
            )

        remaining = max_samples - processed_samples
        if len(states) > remaining:
            states = states[:remaining]
            actions = actions[:remaining]

        action_chunks = _build_action_chunks(actions if len(actions) == expected_length else _load_episode_arrays(episode_file, state_col, action_col)[1], action_horizon)
        action_chunks = action_chunks[: len(states)]
        action_chunks = _apply_delta_actions(states, action_chunks, delta_action_mask)

        buffered_states.append(states)
        buffered_actions.append(action_chunks)
        buffered_count += len(states)
        processed_samples += len(states)
        processed_files += 1

        while buffered_count >= batch_size:
            states_batch = np.concatenate(buffered_states, axis=0)
            actions_batch = np.concatenate(buffered_actions, axis=0)
            stats["state"].update(states_batch[:batch_size])
            stats["actions"].update(actions_batch[:batch_size])

            states_remainder = states_batch[batch_size:]
            actions_remainder = actions_batch[batch_size:]
            buffered_states = [states_remainder] if len(states_remainder) else []
            buffered_actions = [actions_remainder] if len(actions_remainder) else []
            buffered_count = len(states_remainder)

    if buffered_count:
        raise RuntimeError(
            "Internal batching error: sequential processing ended with a non-empty buffer. "
            "The sample count should already be a multiple of batch_size."
        )

    return processed_files, processed_samples


def _process_shuffled_subset(
    *,
    stats: dict[str, normalize.RunningStats],
    episodes: list[dict[str, Any]],
    episode_files: list[Path],
    state_col: str,
    action_col: str,
    action_horizon: int,
    delta_action_mask: np.ndarray | None,
    batch_size: int,
    max_samples: int,
) -> tuple[int, int]:
    import torch

    lengths = np.asarray([int(ep["length"]) for ep in episodes], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    total_frames = int(lengths.sum())
    generator = torch.Generator()
    generator.manual_seed(0)
    selected = torch.randperm(total_frames, generator=generator)[:max_samples].numpy()

    cache = _EpisodeCache(max_size=64, state_col=state_col, action_col=action_col)
    files_seen: set[int] = set()

    for start in tqdm(range(0, len(selected), batch_size), desc="Processing shuffled batches"):
        batch_indices = selected[start : start + batch_size]
        state_batch: list[np.ndarray] = []
        action_batch: list[np.ndarray] = []

        for global_idx in batch_indices:
            ep_pos = int(np.searchsorted(starts, global_idx, side="right") - 1)
            local_idx = int(global_idx - starts[ep_pos])
            episode_index = int(episodes[ep_pos]["episode_index"])
            states, actions = cache.get(episode_index, episode_files[ep_pos])
            files_seen.add(episode_index)

            query_indices = np.minimum(local_idx + np.arange(action_horizon), len(actions) - 1)
            state = states[local_idx]
            action_chunk = actions[query_indices]
            action_chunk = _apply_delta_actions(state[None, :], action_chunk[None, :, :], delta_action_mask)[0]
            state_batch.append(state)
            action_batch.append(action_chunk)

        stats["state"].update(np.stack(state_batch, axis=0))
        stats["actions"].update(np.stack(action_batch, axis=0))

    return len(files_seen), max_samples


def main(
    config_name: str,
    base_dir: str | None = None,
    max_frames: int | None = None,
    state_col: str | None = None,
    action_col: str | None = None,
):
    """Compute norm stats for an EmbodiChain LeRobot dataset.

    Args:
        config_name: OpenPI train config name.
        base_dir: Dataset directory. Defaults to ``$HF_LEROBOT_HOME/<repo_id>``.
        max_frames: Match ``compute_norm_stats.py`` sampling semantics when provided.
        state_col: Optional override for the state parquet column.
        action_col: Optional override for the action parquet column.
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("This fast script only supports local LeRobot parquet datasets, not RLDS.")
    if len(data_config.action_sequence_keys) != 1:
        raise NotImplementedError(
            "This fast script currently supports exactly one action sequence key. "
            f"Got: {data_config.action_sequence_keys}"
        )

    dataset_dir = _resolve_dataset_dir(base_dir, data_config.repo_id)
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"Expected LeRobot metadata under {dataset_dir / 'meta'}")

    info = _read_json(info_path)
    episodes = sorted(_read_jsonl(episodes_path), key=lambda ep: int(ep["episode_index"]))
    state_col, action_col = _detect_columns(info, state_col, action_col)
    episode_files = [_episode_file_path(dataset_dir, info, int(ep["episode_index"])) for ep in episodes]
    _warn_about_extra_parquet(dataset_dir, episode_files)

    total_frames = sum(int(ep["length"]) for ep in episodes)
    meta_total_frames = int(info.get("total_frames", total_frames))
    if total_frames != meta_total_frames:
        raise ValueError(f"episodes.jsonl lengths sum to {total_frames}, but info.json says {meta_total_frames}")

    batch_size = int(config.batch_size)
    if max_frames is not None and max_frames < total_frames:
        num_batches = max_frames // batch_size
        shuffle_subset = True
    else:
        num_batches = total_frames // batch_size
        shuffle_subset = False
    max_samples = num_batches * batch_size

    if max_samples < 2:
        raise ValueError(f"Not enough samples to compute stats after drop_last: {max_samples}")

    delta_action_mask = _get_delta_action_mask(data_config)
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    print(f"Reading dataset: {dataset_dir}")
    print(f"Config: {config_name}")
    print(f"State column: {state_col}")
    print(f"Action column: {action_col}")
    print(f"FPS: {info.get('fps')}")
    print(f"Episodes: {len(episodes)}")
    print(f"Total frames from metadata: {total_frames}")
    print(f"Batch size: {batch_size}")
    print(f"Action horizon: {config.model.action_horizon}")
    print(f"Samples used after drop_last: {max_samples}")
    print(f"DeltaActions enabled: {delta_action_mask is not None}")

    if shuffle_subset:
        files_processed, samples_processed = _process_shuffled_subset(
            stats=stats,
            episodes=episodes,
            episode_files=episode_files,
            state_col=state_col,
            action_col=action_col,
            action_horizon=config.model.action_horizon,
            delta_action_mask=delta_action_mask,
            batch_size=batch_size,
            max_samples=max_samples,
        )
    else:
        files_processed, samples_processed = _process_sequential(
            stats=stats,
            episodes=episodes,
            episode_files=episode_files,
            state_col=state_col,
            action_col=action_col,
            action_horizon=config.model.action_horizon,
            delta_action_mask=delta_action_mask,
            batch_size=batch_size,
            max_samples=max_samples,
        )

    print(f"\nProcessed {files_processed} files with {samples_processed} samples")

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    for key, stat_result in norm_stats.items():
        print(f"\n{key} statistics:")
        print(f"  Shape: {stat_result.mean.shape}")
        print(f"  Mean: {stat_result.mean}")
        print(f"  Std: {stat_result.std}")
        print(f"  Q01: {stat_result.q01}")
        print(f"  Q99: {stat_result.q99}")

    output_path = config.assets_dirs / data_config.repo_id
    print(f"\nWriting stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(f"Normalization stats saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
