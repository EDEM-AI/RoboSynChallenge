"""Regression checks for legacy ACT and Diffusion checkpoint processors."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from safetensors.torch import save_file
import torch

from policy.lerobot_checkpoint import load_processors


class LegacyProcessorsTest(unittest.TestCase):
    def test_image_feature_names_and_action_inverse(self):
        state = "observation.state"
        camera = "observation.images.cam_left_wrist"
        features = {
            state: PolicyFeature(FeatureType.STATE, (2,)),
            camera: PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
        }
        action = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
        for config_type, action_stats, neutral in (
            (ACTConfig, {"mean": 3.0, "std": 2.0}, 3.0),
            (DiffusionConfig, {"min": 1.0, "max": 7.0}, 4.0),
        ):
            with self.subTest(policy=config_type.__name__), TemporaryDirectory() as directory:
                root = Path(directory)
                config = config_type(device="cpu", input_features=features, output_features=action)
                self.assertEqual(
                    config.normalization_mapping["VISUAL"], NormalizationMode.MEAN_STD
                )
                weights = {
                    "normalize_inputs.buffer_observation_images_cam_left_wrist.mean":
                        torch.full((3, 1, 1), 0.2),
                    "normalize_inputs.buffer_observation_images_cam_left_wrist.std":
                        torch.full((3, 1, 1), 0.4),
                }
                for name, value in action_stats.items():
                    weights[f"normalize_inputs.buffer_observation_state.{name}"] = torch.full((2,), value)
                    weights[f"unnormalize_outputs.buffer_action.{name}"] = torch.full((2,), value)
                    weights[f"normalize_targets.buffer_action.{name}"] = torch.full((2,), value + 1)
                save_file(weights, root / "model.safetensors")

                pre, post = load_processors(config, root, "cpu")
                obs = pre({
                    state: torch.full((1, 2), neutral),
                    camera: torch.full((1, 3, 128, 128), 0.6),
                    "action": torch.full((1, 2), neutral + 1),
                })
                torch.testing.assert_close(obs[state], torch.zeros(1, 2))
                torch.testing.assert_close(obs[camera], torch.ones(1, 3, 128, 128))
                torch.testing.assert_close(obs["action"], torch.zeros(1, 2))
                torch.testing.assert_close(post(torch.zeros(1, 2)), torch.full((1, 2), neutral))

                pre.save_pretrained(root)
                post.save_pretrained(root)
                reloaded_pre, reloaded_post = load_processors(config, root, "cpu")
                inputs = {state: torch.full((1, 2), neutral)}
                torch.testing.assert_close(reloaded_pre(inputs)[state], pre(inputs)[state])
                torch.testing.assert_close(
                    reloaded_post(torch.ones(1, 2)), post(torch.ones(1, 2))
                )


if __name__ == "__main__":
    unittest.main()
