# Deploy Your Policy

To deploy and evaluate your policy, you need to **modify the following three files**:

* `eval.sh`: [eval.sh demo](https://github.com/EDEM-AI/RoboSynChallenge/tree/main/policy/Your_Policy/eval.sh)
* `deploy_policy.yml`: [deploy_policy.yml demo](https://github.com/EDEM-AI/RoboSynChallenge/tree/main/policy/Your_Policy/deploy_policy.yml)
* `deploy_policy.py`: [deploy_policy.py demo](https://github.com/EDEM-AI/RoboSynChallenge/tree/main/policy/Your_Policy/deploy_policy.py)

In `deploy_policy.py`, the following components are defined: `get_model` for loading the policy model, `encode_obs` for observation processing, `encode_action` for action processing, `eval` for runing one inference cycle and execute actions in the environment,and `reset_model` for cleaning the model cache at the beginning of every evaluation episode. Additionally, you need to wrap your policy model class to further encapsulate functions such as `set_language`, `update_observation_window`, `get_action`. For details, please refer to [pi0_model](https://github.com/EDEM-AI/RoboSynChallenge/tree/main/policy/pi0/pi0_model.py).

`get_action` along with the control loop that handles observation acquisition and action execution.

The `deploy_policy.yml` file specifies the input parameters. Some of these parameters are model-related and will ultimately be passed as `usr_args` to the `get_model` function to help locate, define, and load your model. The other part is the basic experimental setup.

In `eval.sh`, the parameters specified after `overrides` can be used to overwrite those in `deploy_policy.yml`, allowing you to specify different settings without manually modifying the YAML file each time.

```
# policy/Your_Policy/deploy_policy.py

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
```

---

## 🔧 `deploy_policy.yml`

You are free to **add any parameters** needed in `deploy_policy.yml` to specify your model setup (e.g., checkpoint path, model type, architecture details). The entire YAML content will be passed to `deploy_policy.py` as `usr_args`, which will be available in the `get_model()` function.

```yaml
# Your-Own-Policy Evaluation Configuration

# ------------------------------------------------------------------
# Basic experiment configuration (Each policy must be retained.)
# ------------------------------------------------------------------
policy_name: null # must be modified to your policy name
task_name: null
setting: null
model_name: null
seed: 0
max_episodes: 20
max_steps: 1000 # Maximum environment steps per episode. Effective limit is max(deploy_config.max_steps, gym_config.max_episode_steps).
headless: false
pytorch_device: cuda
filter_dataset_saving: true
eval_video_log: true
eval_video_obs_key: cam_high

# ------------------------------------------------------------------
# Add policy-related parameters you needing, used for get_model()
# ------------------------------------------------------------------
# ...

```

Set `task_name` to one of: `click_bell`, `handle_basket`, `water_pouring`, `table_rearrangement`, `items_handover`, `drawer_open_place`, `mixer_operating`, `item_assembly`, `manipulate_pipette`, `sample_loading`, or `open_pan`.

---

## 🖥️ `eval.sh`

Update the script to pass additional arguments to override default values in `deploy_policy.yml`.

```bash
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

POLICY_NAME=Your_Policy # [TODO]

TASK_NAME="${1}"
SETTING="${2}"
TRAIN_CONFIG="${3}"
MODEL_NAME="${4}"
GPU_ID="${5}"
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

echo "========================================="
echo "  Your-Own-Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  GPU:        $GPU_ID"
echo "========================================="

cd "$REPO_ROOT" # move to RoboSynChallenge root

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --model_name "$MODEL_NAME" \
    # [TODO] add parameters here
```

---

## 🧠  `deploy_policy.py`

You need to implement the following methods in `deploy_policy.py`:

### `encode_action(obs: dict) -> dict`

Optional. This function is used to preprocess the raw policy output (e.g., data format change, etc.)

---

### `encode_obs(obs: dict) -> dict`

Optional. This function is used to preprocess the raw environment observation (e.g., color channel normalization, reshaping, etc.).

---

### `get_model(usr_args: dict) -> Any`

Required. This function receives the full configuration from `deploy_policy.yml` via `usr_args` and must return the initialized model. You can define your own loading logic here, including parsing checkpoints and network parameters.

---

### `eval(env, model, observation) -> Any`

Required. The main evaluation loop. Given the current environment instance, model, and observation (as a dictionary), this function must compute the next action and execute it in the environment.

---

### `update_obs(obs: dict) -> None`

Optional. Used to update any internal state of the model or observation buffer. Useful if your model requires a history of frames or a memory-based context.

---

### `reset_model() -> None`

Optional but **recommended**. This function is called before the evaluation of **each episode**, allowing you to reset model states such as recurrent memory, history buffers, or context encodings.

---


## ✔️ Run `eval.sh`

```
bash eval.sh ...(input parameters you define)
```

## 📌 Notes
* Your policy should be compatible with the input/output format expected by the simulator.
