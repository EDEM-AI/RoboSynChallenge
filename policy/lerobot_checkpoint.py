"""Load saved LeRobot processors and run policy inference."""

from pathlib import Path

from lerobot.policies.factory import make_pre_post_processors


def load_processors(config, checkpoint_path, device):
    root = Path(checkpoint_path)
    if not all((root / name).is_file() for name in (
        "policy_preprocessor.json", "policy_postprocessor.json"
    )):
        raise FileNotFoundError(
            f"Expected LeRobot preprocessor and postprocessor configs in {root}."
        )
    return make_pre_post_processors(
        config,
        pretrained_path=str(root),
        preprocessor_overrides={"device_processor": {"device": device}},
    )


def select_action(model, batch):
    preprocess, postprocess = model.checkpoint_processors
    return postprocess(model.select_action(preprocess(batch)))


def reset_model(model):
    model.reset()
    for processor in model.checkpoint_processors:
        processor.reset()
