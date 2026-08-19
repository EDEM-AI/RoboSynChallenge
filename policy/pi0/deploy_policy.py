# ----------------------------------------------------------------------------
# π₀ Policy Adapter for RoboSynChallenge
#
# 遵循 RoboTwin 统一评估接口:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import os
import sys
import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.insert(0, parent_directory)

from pi_model import PI0
from policy.inference_timing import timed_inference


def _as_done(value):
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def encode_action(action, env):
    """Convert pi0 output into the torch action format EmbodiChain accepts."""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.shape[0] < env_action_dim:
        raise ValueError(
            f"Policy action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
        )
    action_array = action_array[:env_action_dim]

    action_tensor = torch.as_tensor(
        action_array, dtype=torch.float32, device=env.unwrapped.device
    )
    return action_tensor.unsqueeze(0)


def encode_obs(obs):
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
    img_front_raw = obs["sensor"]["cam_high"]["color"]
    img_left_raw = obs["sensor"]["cam_left_wrist"]["color"]
    img_right_raw = obs["sensor"]["cam_right_wrist"]["color"]

    img_front = img_front_raw[0, ..., :3]
    img_left = img_left_raw[0, ..., :3]
    img_right = img_right_raw[0, ..., :3]

    # Joint state — (num_envs, num_joints) -> squeeze env dim
    state = obs["robot"]["qpos"][0]
    img_arr = [img_front, img_right, img_left]

    return img_arr, state


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


def _observation_to_env_actions(env, model, obs):
    """Convert one raw observation into an executable action chunk."""
    img_arr, state = encode_obs(obs)
    model.update_observation_window(img_arr, state)
    actions = model.get_action()[: model.pi0_step]
    return [encode_action(action, env) for action in actions]


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
        model.set_language(instruction)

    # Time raw observation processing, policy inference, and action formatting.
    action_tensors, inference_time_s = timed_inference(
        _observation_to_env_actions,
        env,
        model,
        obs,
        device=model.pytorch_device,
    )

    # Execute actions one by one in the environment
    final_obs = obs
    info = None
    truncated = False
    for action_tensor in action_tensors:
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        # The `gym_config` setting configures the `actionmanager` to support delta action input;
        # the default action must be absolute `qpos`.
        if env.get_wrapper_attr("is_task_success")():
            break
        if _as_done(truncated):
            break
        # Update observation window after each step
        img_arr, state = encode_obs(final_obs)
        model.update_observation_window(img_arr, state)

    return final_obs, info, truncated, {
        "inference_times_s": [inference_time_s],
        "inference_timing_scope": "raw_observation_to_env_actions",
    }


def reset_model(model):
    """Reset π₀ internal state (observation window and instruction)."""
    model.reset_obsrvationwindows()
