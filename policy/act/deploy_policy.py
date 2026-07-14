# ----------------------------------------------------------------------------
# LeRobot ACT Policy Adapter for RoboSynChallenge
#
# Follows the same unified evaluation interface as policy/pi0:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info, truncated
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

def get_model(usr_args):
    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in usr_args.")

    device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
    cli_overrides = [f"--device={device}"]
    try:
        policy = ACTPolicy.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
    except TypeError as exc:
        if "cli_overrides" not in str(exc):
            raise
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
        policy = ACTPolicy.from_pretrained(checkpoint_path, config=config)
    policy.eval()

    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(usr_args.get("image_key_map") or {})
    image_keys = list(policy.config.image_features.keys())
    for image_key in image_keys:
        image_key_map.setdefault(
            image_key,
            image_key.removeprefix("observation.images."),
        )

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 8))
    policy.state_obs_path = usr_args.get("state_obs_path", "robot/qpos")
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.act_image_keys = image_keys
    policy.image_key_map = image_key_map

    if policy.act_step <= 0:
        raise ValueError(f"act_step must be positive, got {policy.act_step}.")

    return policy

def eval(env, model, obs):
    final_obs = obs
    info = None
    truncated = False

    for _ in range(model.act_step):
        state = final_obs
        for key in str(model.state_obs_path).split("/"):
            if key:
                state = state[key]
        state_tensor = state.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=torch.float32, device=model.act_device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        batch = {"observation.state": state_tensor}
        for image_key in model.act_image_keys:
            camera_name = model.image_key_map.get(
                image_key,
                image_key.removeprefix("observation.images."),
            )
            image = final_obs["sensor"][camera_name]["color"]
            image_tensor = image.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(image, torch.Tensor) else torch.as_tensor(image, dtype=torch.float32, device=model.act_device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
            if torch.max(image_tensor) > 1.5:
                image_tensor = image_tensor / 255.0
            batch[image_key] = image_tensor

        action = model.select_action(batch)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(f"Expected policy action shape [B, D], but got {tuple(action.shape)}.")

        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        policy_action_dim = int(action.shape[-1])
        if policy_action_dim != env_action_dim:
            message = f"Policy action has dim {policy_action_dim}, but env expects {env_action_dim}."
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        action_tensor = action.detach().to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if isinstance(truncated, torch.Tensor):
            is_truncated = truncated.any().item()
        elif isinstance(truncated, np.ndarray):
            is_truncated = truncated.any()
        else:
            is_truncated = bool(truncated)
        if is_truncated:
            break

    return final_obs, info, truncated


def reset_model(model):
    model.reset()
