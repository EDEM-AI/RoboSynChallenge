"""Load LeRobot processors from new checkpoints or legacy embedded statistics."""

import logging
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.policies.factory import make_pre_post_processors
from safetensors import safe_open

logger = logging.getLogger(__name__)

_STAT_NAMES = {
    NormalizationMode.MEAN_STD: ("mean", "std"),
    NormalizationMode.MIN_MAX: ("min", "max"),
    NormalizationMode.QUANTILES: ("q01", "q99"),
    NormalizationMode.QUANTILE10: ("q10", "q90"),
}


def _read_stats(weights, prefix, features, norm_map):
    stats = {}
    for name, feature in features.items():
        mode = norm_map.get(feature.type, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY:
            continue

        # Legacy buffer names escape dots. Use the configured feature name to
        # preserve literal underscores, for example in cam_left_wrist.
        buffer = name.replace(".", "_")
        shapes = {tuple(feature.shape)}
        if feature.type == FeatureType.VISUAL:
            channels = feature.shape[0]
            shapes.update({(channels,), (channels, 1, 1)})
        stats[name] = {}
        for statistic in _STAT_NAMES[mode]:
            key = f"{prefix}.buffer_{buffer}.{statistic}"
            if key not in weights.keys():
                raise ValueError(f"Checkpoint is missing {key}.")
            value = weights.get_tensor(key)
            if tuple(value.shape) not in shapes or not value.isfinite().all():
                raise ValueError(f"Checkpoint contains invalid normalization statistics for {key}.")
            stats[name][statistic] = value
    return stats


def load_processors(config, checkpoint_path, device):
    root = Path(checkpoint_path)
    pre_file = root / "policy_preprocessor.json"
    post_file = root / "policy_postprocessor.json"
    if pre_file.is_file() or post_file.is_file():
        if not (pre_file.is_file() and post_file.is_file()):
            raise FileNotFoundError(f"Expected both LeRobot processor configs in {root}.")
        return make_pre_post_processors(
            config,
            pretrained_path=str(root),
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    norm_map = config.normalization_mapping
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as weights:
        inputs = _read_stats(weights, "normalize_inputs", config.input_features, norm_map)
        outputs = _read_stats(weights, "unnormalize_outputs", config.output_features, norm_map)
        has_targets = any(key.startswith("normalize_targets.buffer_") for key in weights.keys())
        targets = (
            _read_stats(weights, "normalize_targets", config.output_features, norm_map)
            if has_targets
            else outputs
        )

    preprocess, _ = make_pre_post_processors(config, dataset_stats={**inputs, **targets})
    _, postprocess = make_pre_post_processors(config, dataset_stats=outputs)
    logger.info("Loaded embedded LeRobot normalization statistics from %s", root)
    return preprocess, postprocess


def select_action(model, batch):
    preprocess, postprocess = model.checkpoint_processors
    return postprocess(model.select_action(preprocess(batch)))


def reset_model(model):
    model.reset()
    for processor in model.checkpoint_processors:
        processor.reset()
