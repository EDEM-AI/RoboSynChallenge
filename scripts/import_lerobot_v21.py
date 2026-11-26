"""Import a local legacy LeRobot dataset into a new v3.0 directory."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import convert_dataset_v21_to_v30 as converter
import pyarrow.parquet as pq


def feature_aliases(features):
    candidates = {"observation.qpos": "observation.state"}
    for camera in ("cam_high", "cam_right_wrist", "cam_left_wrist"):
        candidates[f"{camera}.color"] = f"observation.images.{camera}"
    return {
        old: new
        for old, new in candidates.items()
        if old in features and new not in features
    }


def stage_metadata(source, staging, info, aliases):
    """Present renamed camera and stats keys to the official metadata converter."""
    meta = staging / "meta"
    meta.mkdir(parents=True)
    info["features"] = {
        aliases.get(key, key): value for key, value in info["features"].items()
    }
    (meta / "info.json").write_text(json.dumps(info))
    for name in ("tasks.jsonl", "episodes.jsonl"):
        (meta / name).symlink_to(source / "meta" / name)
    with (meta / "episodes_stats.jsonl").open("w") as output:
        for row in converter.load_jsonlines(source / "meta/episodes_stats.jsonl"):
            row["stats"] = {
                aliases.get(key, key): value for key, value in row["stats"].items()
            }
            output.write(json.dumps(row) + "\n")
    for video in sorted((source / "videos").glob("*/*/*.mp4")):
        chunk, camera, name = video.relative_to(source / "videos").parts
        destination = staging / "videos" / chunk / aliases.get(camera, camera) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(video)


def convert_arrow_data(source, destination, aliases, max_size_mb):
    """Write v3.0 shards without converting Arrow extension arrays to pandas."""
    chunk = 0
    file = 0
    writer = None
    shard_size = 0.0
    frame_index = 0
    episodes = []
    try:
        for episode_index, path in enumerate(
            sorted((source / "data").glob("*/*.parquet"))
        ):
            size = converter.get_parquet_file_size_in_mb(path)
            if writer is not None and shard_size + size >= max_size_mb:
                writer.close()
                writer = None
                shard_size = 0.0
                chunk, file = converter.update_chunk_file_indices(
                    chunk, file, converter.DEFAULT_CHUNK_SIZE
                )
            table = pq.read_table(path)
            aliases_in_table = {
                old: new for old, new in aliases.items() if old in table.column_names
            }
            if aliases_in_table:
                features = Features.from_arrow_schema(table.schema)
                features = Features(
                    {aliases.get(key, key): value for key, value in features.items()}
                )
                table = table.rename_columns(list(features)).replace_schema_metadata(
                    features.arrow_schema.metadata
                )
            if writer is None:
                output_path = destination / converter.DEFAULT_DATA_PATH.format(
                    chunk_index=chunk, file_index=file
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            end_index = frame_index + table.num_rows
            episodes.append(
                {
                    "episode_index": episode_index,
                    "data/chunk_index": chunk,
                    "data/file_index": file,
                    "dataset_from_index": frame_index,
                    "dataset_to_index": end_index,
                }
            )
            shard_size += size
            frame_index = end_index
    finally:
        if writer is not None:
            writer.close()
    return episodes


def import_dataset(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source):
        raise ValueError("Choose an output outside the original dataset.")
    converter.validate_local_dataset_version(source)
    info = json.loads((source / "meta/info.json").read_text())
    aliases = feature_aliases(info["features"])
    data_size = converter.DEFAULT_DATA_FILE_SIZE_IN_MB
    video_size = converter.DEFAULT_VIDEO_FILE_SIZE_IN_MB
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".lerobot-import-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        metadata_source = source
        if aliases:
            metadata_source = staging / "metadata_source"
            stage_metadata(source, metadata_source, info, aliases)
        output = staging / "dataset"
        converter.convert_info(metadata_source, output, data_size, video_size)
        converter.convert_tasks(metadata_source, output)
        episodes = convert_arrow_data(source, output, aliases, data_size)
        videos = converter.convert_videos(metadata_source, output, video_size)
        converter.convert_episodes_metadata(metadata_source, output, episodes, videos)
        dataset = LeRobotDataset("local/imported", root=output, video_backend="pyav")
        if (
            dataset.num_episodes != info["total_episodes"]
            or len(dataset) != info["total_frames"]
        ):
            raise ValueError(
                "Converted dataset disagrees with the original episode or frame count."
            )
        if destination.exists():
            raise FileExistsError(destination)
        output.rename(destination)
    print(f"Imported {info['total_episodes']} episodes into {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
