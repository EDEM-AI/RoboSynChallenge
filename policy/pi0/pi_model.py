"""OpenPI inference for RoboSynChallenge using its original checkpoint layout."""

import dataclasses
from pathlib import Path

from openpi.policies import policy_config
from openpi.training import config as openpi_config


class PI0:
    policy_name = "pi0"

    def __init__(
        self,
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        pytorch_device="cuda",
        *,
        checkpoint_path=None,
        asset_id=None,
    ):
        self.pytorch_device = pytorch_device
        self.pi0_step = pi0_step
        self.observation_window = None
        self.instruction = None
        checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path
            else (
                Path(__file__).resolve().parents[1]
                / self.policy_name
                / "checkpoints"
                / train_config_name
                / model_name
                / str(checkpoint_id)
            )
        )
        config = openpi_config.get_config(train_config_name)
        if pytorch_device == "cpu" and hasattr(config.model, "pytorch_compile_mode"):
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, pytorch_compile_mode=None),
            )
        assets = config.data.assets
        default_asset = assets.asset_id or config.data.repo_id
        if (
            asset_id is None
            and not (
                checkpoint / "assets" / str(default_asset) / "norm_stats.json"
            ).is_file()
        ):
            matches = list((checkpoint / "assets").rglob("norm_stats.json"))
            if len(matches) == 1:
                asset_id = str(matches[0].parent.relative_to(checkpoint / "assets"))
        if asset_id is not None:
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(
                    config.data, assets=dataclasses.replace(assets, asset_id=asset_id)
                ),
            )
        self.policy = policy_config.create_trained_policy(
            config, checkpoint, pytorch_device=pytorch_device
        )

    def set_language(self, instruction):
        self.instruction = instruction

    def update_observation_window(self, images, state):
        front, right_wrist, left_wrist = images
        self.observation_window = {
            "observation/image": front,
            "observation/left_wrist_image": left_wrist,
            "observation/right_wrist_image": right_wrist,
            "observation/state": state,
            "prompt": self.instruction,
        }

    def get_action(self):
        if self.observation_window is None:
            raise RuntimeError(
                "Update the observation window before requesting an action."
            )
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
