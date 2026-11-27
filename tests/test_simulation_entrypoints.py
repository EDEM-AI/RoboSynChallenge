"""CPU regressions for policy evaluation and recorded replay entry points."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs import demo
from embodichain.lab.gym.utils import registration
from scripts import eval_policy, replay_seeded


class BudgetEnv:
    def __init__(self, max_episode_steps):
        self.unwrapped = self
        self.cfg = SimpleNamespace(task_program=None)
        self.max_episode_steps = max_episode_steps
        self.steps = 0

    def reset(self, *, seed, options):
        self.steps = 0
        return {}, {}

    def step(self, action):
        self.steps += 1
        return (
            {}, 0, torch.tensor([False]),
            torch.tensor([self.steps >= self.max_episode_steps]),
            {"elapsed_steps": torch.tensor([self.steps])},
        )

    def get_wrapper_attr(self, name):
        return getattr(self, name)

    def is_task_success(self):
        return False

    def close(self):
        pass


@pytest.mark.parametrize("policy_limit", [None, 3])
def test_policy_step_limit_preserves_expert_seed_eligibility(
    monkeypatch, tmp_path, policy_limit
):
    config = {
        "policy_name": "test_policy",
        "task_name": "other_tasks/manipulate_pipette_two_beaker",
        "setting": "clear",
        "max_episodes": 1,
        "eval_fixed_episode_seed": 19,
        "eval_max_steps": policy_limit,
        "eval_result_dir": str(tmp_path),
    }

    def run_expert(env):
        for _ in range(4):
            _, _, _, truncated, _ = env.step(None)
            if truncated.any():
                return SimpleNamespace(
                    completed=False, success=[False], terminal_reason="truncated",
                    length=env.steps,
                )
        return SimpleNamespace(
            completed=True, success=[True], terminal_reason="success", length=env.steps,
        )

    def run_policy_chunk(env, model, obs):
        for _ in range(10):
            obs, _, _, truncated, info = env.step(None)
            if truncated.any():
                break
        rollout_steps.append(int(info["elapsed_steps"].item()))
        return obs, info, truncated, []

    rollout_steps = []
    policy = SimpleNamespace(
        get_model=lambda config: None,
        reset_model=lambda model: None,
        eval=run_policy_chunk,
    )
    monkeypatch.setattr(eval_policy, "parse_args_and_config", lambda: config)
    monkeypatch.setattr(eval_policy, "load_policy_adapter", lambda name: policy)
    monkeypatch.setattr(registration, "discover_task_packages", lambda: None)
    monkeypatch.setattr(registration, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        eval_policy.gym_utils,
        "build_env_cfg_from_args",
        lambda args: (object(), {"id": "test", "max_episode_steps": 5}, None),
    )
    monkeypatch.setattr(
        eval_policy.gym, "make",
        lambda id, cfg, max_episode_steps=5: BudgetEnv(max_episode_steps),
    )
    monkeypatch.setattr(eval_policy, "suppress_expert_output", nullcontext)
    monkeypatch.setattr(demo, "execute_demo_episode", run_expert)

    eval_policy.main()

    (result_path,) = tmp_path.rglob("evaluation_metrics.json")
    result = json.loads(result_path.read_text())
    assert result["feasibility_filter"]["skipped_seed_count"] == 0
    assert result["episodes"][0]["seed"] == 19
    assert result["episodes"][0]["expert_plan_steps"] == 4
    assert rollout_steps == [policy_limit or 5]


@pytest.mark.parametrize(
    "category, uid",
    [("rigid_objects", "beaker"), ("articulations", "pipette"), ("cables", "wire")],
)
@pytest.mark.parametrize("recorded_reset", [False, True])
def test_replay_rejects_missing_entities_before_state_access(
    monkeypatch, category, uid, recorded_reset
):
    expected = TensorDict(
        {category: {uid: {"pose": torch.zeros(1, 7)}}}, batch_size=[1]
    )
    env = SimpleNamespace(
        sim=SimpleNamespace(
            get_rigid_object_uid_list=lambda: [],
            get_articulation_uid_list=lambda: [],
            get_cable_object_uid_list=lambda: [],
        )
    )

    def unexpected_state_access(*args):
        pytest.fail("The scene must be validated before accessing simulator state")

    monkeypatch.setattr(replay_seeded, "restore_trajectory_state", unexpected_state_access)
    monkeypatch.setattr(replay_seeded, "capture_trajectory_state", unexpected_state_access)

    with pytest.raises(ValueError, match=f"missing recorded {category}.*{uid}"):
        replay_seeded.verify_initial_state(env, expected, recorded_reset=recorded_reset)


@pytest.mark.parametrize(
    "module_name, env_name, adapter_name, program_name, segment_name, expected_motion",
    [
        (
            "carry_pour_programs",
            "HandleBasket",
            "_BASKET_ADAPTER",
            "_BASKET_PROGRAM",
            "clear_left_arm",
            ((95, 2.0), (50, 2.0)),
        ),
        (
            "pipette_program",
            "ManipulatePipette",
            "_ADAPTER",
            "_PROGRAM",
            "press_plunger",
            ((50, 2.0), (50, 0.8)),
        ),
        (
            "placement_programs",
            "SampleLoading",
            "_TUBE_ADAPTER",
            "_TUBE_PROGRAM",
            "hand_over_test_tube",
            ((60, 1.2), (98, 1.2)),
        ),
    ],
)
def test_expert_bridge_selects_motion_preset_for_one_occurrence(
    module_name, env_name, adapter_name, program_name, segment_name, expected_motion
):
    from importlib import import_module

    from embodichain.lab.task_program.integrations.invocation_policy import (
        BoundInvocationPolicies,
    )

    module = import_module(f"robosynchallenge.{module_name}")
    env_type = getattr(module, env_name)
    registration = getattr(module, adapter_name).registration
    program = registration.catalog.preflight(getattr(module, program_name))
    segment = next(s for s in program.iter_segments() if s.name == segment_name)
    calls = []

    def create_bridge(compiled, *, invocation_policy_bindings, **kwargs):
        BoundInvocationPolicies(compiled, invocation_policy_bindings)
        calls.extend(invocation_policy_bindings)
        return object()

    env = SimpleNamespace(
        cfg=SimpleNamespace(task_program_execution="expert"),
        num_envs=1,
        call_motion_presets=env_type.call_motion_presets,
        _checked_task_program_adapter=lambda: SimpleNamespace(
            create_bridge=create_bridge
        ),
    )
    env_type.create_task_program_bridge(env, program)

    assert len(calls) == 1
    assert calls[0].invocation_id == f"{program.program_id}/{segment.segment_id}:0"
    policy_by_id = {
        preset.preset_id: preset.motion_policy
        for preset in registration.robot_profile_binding.presets
    }
    selected = (
        policy_by_id[program.integration.runtime_preset],
        policy_by_id[calls[0].preset_id],
    )
    assert (
        tuple((p.sample_count, p.acceleration_limit) for p in selected)
        == expected_motion
    )


def test_cutlery_clearance_checks_interior_samples():
    from robosynchallenge.cutlery_parallel import _lanes_are_clear

    left_low = torch.zeros(3, 1, 3)
    right_low = torch.tensor([[[0.2, 0.0, 0.0]]]).repeat(3, 1, 1)
    left = (left_low, left_low + 0.1)
    assert _lanes_are_clear(left, (right_low, right_low + 0.1))

    right_low[1, 0, 0] = 0.09
    assert not _lanes_are_clear(left, (right_low, right_low + 0.1))


def test_two_beaker_pipette_uses_its_registered_policy(monkeypatch):
    from embodichain.lab.gym.envs import EmbodiedEnv

    from robosynchallenge.experimental_programs import (
        _TRANSFER_ADAPTER,
        _TRANSFER_PROGRAM,
        TransferPipette,
    )

    program = _TRANSFER_ADAPTER.registration.catalog.preflight(_TRANSFER_PROGRAM)
    env = object.__new__(TransferPipette)
    sdk_bridge = object()

    def create_sdk_bridge(self, compiled, **kwargs):
        assert compiled is program
        return sdk_bridge

    monkeypatch.setattr(EmbodiedEnv, "create_task_program_bridge", create_sdk_bridge)
    assert env.create_task_program_bridge(program) is sdk_bridge
