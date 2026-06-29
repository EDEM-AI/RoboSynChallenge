# import packages and module here
import numpy as np
def encode_action(action, env):
    """
    Convert Your-Own-Policy output into the torch action format EmbodiChain accepts.
    Refer to https://github.com/EDEM-AI/RoboSynChallenge/tree/main/policy/pi0/deploy_policy.py for an example implementation.
    """
    actions = action
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))

    # ...
    return actions

def encode_obs(observation):  # Post-Process Observation
    """
    Convert gym Gymnasium Dict observation to Your-Own-Policy input format.
    """
    obs = observation
    # ...
    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    """
    Create and return a policy model instance.
    """
    Your_Model = None
    # ...
    return Your_Model  # return your policy model


def eval(env, model, obs):
    """Run one inference cycle and execute actions in the environment.

    This function:
    1. Sets the language instruction (on first call when observation_window is None)
    2. Encodes observation and updates the model's observation window
    3. Calls model.get_action() to get multi-step actions
    4. Steps through each action in the environment
    """
    # Set language instruction if first call (Try to keep it unchanged)
    # implement the `set_language` function in your own policy object.
    if model.observation_window is None:
        instruction = getattr(env, "_current_instruction", None)
        model.set_language(instruction)

    # Encode and update observation window
    obs = encode_obs(obs)
    model.update_observation_window(obs)
    # implement the `update_observation_window` function in your own policy object.


    # Get multi-step actions from Your-Own-Policy
    actions = model.get_action()
    # implement the `get_action` function in your own policy object.

    # Execute actions one by one in the environment
    for action in actions:
        action_tensor = encode_action(action, env) # Map the actions output by your model to the format required by EmbodiChain.
        observation, reward, terminated, truncated, info = env.step(action_tensor)
        # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        # Absolute joint control is the default;
        # if other control modes—such as relative endpose control are required, you must add an `actions` field to the `gym_config` for the specific task to utilize the action manager.
        # Please refer to https://dexforce.github.io/EmbodiChain/main/overview/gym/action_functors.html for details.

        if truncated.any():
            break

        # Update observation window after each step
        obs = encode_obs(observation)
        model.update_observation_window(obs)

    return observation, info, truncated

def reset_model(model):
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass
