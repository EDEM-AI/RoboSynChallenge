"""Merge v3.0 recordings while retaining EmbodiChain segment annotations."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from embodichain.data_pipeline.datasets.inspection import validate_dataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def merge(sources: list[Path], destination: Path, repo_id: str) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    datasets = []
    sidecar_rows = []
    subtasks = None
    episode_offset = 0
    for index, source in enumerate(sources):
        info = json.loads((source / "meta/info.json").read_text())
        if info["codebase_version"] != "v3.0" or info["total_episodes"] < 1:
            raise ValueError(f"Expected a nonempty v3.0 dataset at {source}.")
        dataset = LeRobotDataset(
            f"local/source_{index}", root=source, video_backend="pyav"
        )
        source_subtasks = pq.read_table(source / "meta/subtasks.parquet")
        if subtasks is not None and not source_subtasks.equals(subtasks):
            raise ValueError(f"Subtask labels or indices differ in {source}.")
        subtasks = source_subtasks
        with (source / "meta/embodichain_episodes.jsonl").open() as sidecar:
            rows = [json.loads(line) for line in sidecar if line.strip()]
        if len(rows) != dataset.num_episodes:
            raise ValueError(f"Episode sidecar is incomplete in {source}.")
        for row in rows:
            for key in ("episode_index", "lerobot_episode_index"):
                if key in row:
                    row[key] += episode_offset
        sidecar_rows.extend(rows)
        episode_offset += dataset.num_episodes
        datasets.append(dataset)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".lerobot-merge-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "dataset"
        merge_datasets(datasets, repo_id, output_dir=staging)
        shutil.copyfile(
            sources[0] / "meta/subtasks.parquet", staging / "meta/subtasks.parquet"
        )
        with (staging / "meta/embodichain_episodes.jsonl").open("w") as sidecar:
            for row in sidecar_rows:
                sidecar.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = validate_dataset(staging)
        if not report.ok:
            raise ValueError(json.dumps(report.to_dict(), ensure_ascii=False))
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    print(f"Merged {episode_offset} episodes into {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "sources", type=Path, nargs="+", help="Finalized EmbodiChain recording roots"
    )
    args = parser.parse_args()
    merge(args.sources, args.output, args.repo_id)


if __name__ == "__main__":
    main()
