# ----------------------------------------------------------------------------
# π₀ Policy Adapter for RoboSynChallenge
#
# 遵循 RoboTwin 统一评估接口:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs 或 (obs, diagnostics)
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import os
import sys
import numpy as np
import json
import torch

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.insert(0, parent_directory)

from pi_model import PI0


def _unwrap_env(env):
    return getattr(env, "unwrapped", env)


def _get_env_action_dim(env):
    """Return the flat action dimension expected by the EmbodiChain env."""
    raw_env = _unwrap_env(env)
    action_space = getattr(raw_env, "single_action_space", None)
    if action_space is None:
        action_space = getattr(env, "action_space", None)

    shape = getattr(action_space, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(np.prod(shape))


def _get_env_device(env):
    return getattr(_unwrap_env(env), "device", torch.device("cpu"))


def _get_embodichain_camera_color(obs, camera_uid):
    try:
        return obs["sensor"][camera_uid]["color"]
    except Exception as exc:
        raise KeyError(
            f"EmbodiChain camera color observation 'sensor/{camera_uid}/color' not found."
        ) from exc


def _get_embodichain_obs_value(obs, key):
    try:
        return obs[key]
    except Exception:
        pass

    value = obs
    for part in key.split("/"):
        try:
            value = value[part]
        except Exception as exc:
            raise KeyError(f"EmbodiChain observation path '{key}' not found.") from exc
    return value


def _format_rgb_image(image):
    image = _to_numpy(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image shape (H, W, 3), got {image.shape}.")
    if image.dtype != np.uint8:
        if image.size and np.issubdtype(image.dtype, np.floating) and np.max(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _any_bool(value):
    return bool(np.asarray(_to_numpy(value)).any())


def _format_env_action(action, env):
    """Convert pi0 output into the torch action format EmbodiChain accepts."""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = _get_env_action_dim(env)
    if env_action_dim is not None:
        if action_array.shape[0] < env_action_dim:
            raise ValueError(
                f"Policy action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
            )
        action_array = action_array[:env_action_dim]

    action_tensor = torch.as_tensor(
        action_array, dtype=torch.float32, device=_get_env_device(env)
    )
    return action_tensor.unsqueeze(0)


def encode_obs(obs, env, model):
    """Convert gym Gymnasium Dict observation to π₀ input format.

    EmbodiChain observation keys:
        "sensor/cam_high/color"        -> base camera
        "sensor/cam_left_wrist/color"  -> left wrist
        "sensor/cam_right_wrist/color" -> right wrist
        "robot/qpos"                   -> joint state

    Returns:
        img_arr:  list of [img_front, img_right, img_left] as (H, W, C) numpy arrays
        state:    joint state vector
    """
    img_front_raw = _get_embodichain_camera_color(obs, "cam_high")
    img_left_raw = _get_embodichain_camera_color(obs, "cam_left_wrist")
    img_right_raw = _get_embodichain_camera_color(obs, "cam_right_wrist")
    image_available = {
        "cam_high": True,
        "cam_left_wrist": True,
        "cam_right_wrist": True,
    }
    img_front = _format_rgb_image(img_front_raw)
    img_left = _format_rgb_image(img_left_raw)
    img_right = _format_rgb_image(img_right_raw)

    # Joint state — (num_envs, num_joints) -> squeeze env dim
    qpos = _to_numpy(_get_embodichain_obs_value(obs, "robot/qpos"))
    if qpos.ndim > 1:
        qpos = qpos[0]

    state = qpos.astype(np.float32)
    img_arr = [img_front, img_right, img_left]
    input_debug = {
        "image_available": image_available,
        "image_shapes": {
            "cam_high": tuple(img_front.shape),
            "cam_right_wrist": tuple(img_right.shape),
            "cam_left_wrist": tuple(img_left.shape),
        },
        "state_shape": tuple(state.shape),
    }
    return img_arr, state, input_debug


def get_model(usr_args):
    """Create and return a π₀ policy model instance.

    usr_args 中需要的字段:
        train_config_name  — openpi training config name
        model_name         — model name (e.g. "pi0_base")
        checkpoint_id      — checkpoint step number
        pi0_step           — number of action steps to execute per inference (default 50)
    """
    train_config_name = usr_args.get("train_config_name")
    model_name = usr_args.get("model_name")
    checkpoint_id = int(usr_args.get("checkpoint_id", 30000))
    pi0_step = int(usr_args.get("pi0_step", 50))
    pytorch_device = usr_args.get("pytorch_device", "cuda")

    if train_config_name is None or model_name is None:
        raise ValueError(
            "train_config_name and model_name must be provided in usr_args"
        )

    model = PI0(
        train_config_name=train_config_name,
        model_name=model_name,
        checkpoint_id=checkpoint_id,
        pi0_step=pi0_step,
        pytorch_device=pytorch_device,
    )
    return model


def eval(env, model, obs):
    """Run one inference cycle and execute actions in the environment.

    This function:
    1. Sets the language instruction (on first call when observation_window is None)
    2. Encodes observation and updates the model's observation window
    3. Calls model.get_action() to get multi-step actions
    4. Steps through each action in the environment
    """
    # Set language instruction if first call
    if model.observation_window is None:
        instruction = getattr(env, "_current_instruction", None)
        if instruction is None:
            instruction = "do the task"
        model.set_language(instruction)

    # Encode and update observation window
    img_arr, state, input_debug = encode_obs(obs, env, model)
    model.update_observation_window(img_arr, state)

    # Get multi-step actions from π₀
    actions = _to_numpy(model.get_action())[: model.pi0_step]
    diagnostics = {
        "state": _to_numpy(state).copy(),
        "actions": _to_numpy(actions).copy(),
        "instruction": getattr(model, "instruction", None),
        **input_debug,
    }

    # Execute actions one by one in the environment
    final_obs = obs
    env_steps_executed = 0
    last_terminated = False
    last_truncated = False
    for action in actions:
        action_tensor = _format_env_action(action, env)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        env_steps_executed += 1
        last_terminated = _any_bool(terminated)
        last_truncated = _any_bool(truncated)

        # Update observation window after each step
        img_arr, state, _ = encode_obs(final_obs, env, model)
        model.update_observation_window(img_arr, state)

        if last_terminated or last_truncated:
            break

    diagnostics["env_steps_executed"] = env_steps_executed
    diagnostics["terminated"] = last_terminated
    diagnostics["truncated"] = last_truncated

    return final_obs, diagnostics


def reset_model(model):
    """Reset π₀ internal state (observation window and instruction)."""
    model.reset_obsrvationwindows()
