# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.envs.registration import registry

from embodichain.data import get_data_path
from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
    get_func_tag,
)
from robosynchallenge.tasks.atomic_edge_replay import AtomicEdgeReplayActionBankMixin
from robosynchallenge.tasks.atomic_edge_replay import (
    ExplicitTcpAtomicReplayActionBankMixin,
)


TASK_ACTION_BANKS = [
    (
        "table_rearrangement",
        "configs/table_rearrangement/action_config.json",
        "robosynchallenge.tasks.table_rearrangement.action_bank",
        "TableRearrangementActionBank",
    ),
    (
        "click_bell",
        "configs/click_bell/action_config.json",
        "robosynchallenge.tasks.click_bell.action_bank",
        "ClickBellActionBank",
    ),
    (
        "drawer_open_place",
        "configs/drawer_open_place/action_config.json",
        "robosynchallenge.tasks.drawer_open_place.action_bank",
        "DrawerOpenPlaceActionBank",
    ),
    (
        "handle_basket",
        "configs/handle_basket/action_config.json",
        "robosynchallenge.tasks.handle_basket.action_bank",
        "HandleBasketActionBank",
    ),
    (
        "item_assembly",
        "configs/item_assembly/action_config.json",
        "robosynchallenge.tasks.item_assembly.action_bank",
        "ItemAssemblyActionBank",
    ),
    (
        "items_handover",
        "configs/items_handover/action_config.json",
        "robosynchallenge.tasks.items_handover.action_bank",
        "ItemsHandoverActionBank",
    ),
    (
        "manipulate_pipette",
        "configs/manipulate_pipette/action_config.json",
        "robosynchallenge.tasks.manipulate_pipette.action_bank",
        "ManipulatePipetteActionBank",
    ),
    (
        "mixer_operating",
        "configs/mixer_operating/action_config.json",
        "robosynchallenge.tasks.mixer_operating.action_bank",
        "MixerOperatingActionBank",
    ),
    (
        "sample_loading",
        "configs/sample_loading/action_config.json",
        "robosynchallenge.tasks.sample_loading.action_bank",
        "SampleLoadingActionBank",
    ),
    (
        "water_pouring",
        "configs/water_pouring/action_config.json",
        "robosynchallenge.tasks.water_pouring.action_bank",
        "WaterPouringActionBank",
    ),
    (
        "open_pan",
        "configs/other_tasks/open_pan/action_config.json",
        "robosynchallenge.tasks._other_tasks.open_pan.action_bank",
        "OpenPanPickandPlaceActionBank",
    ),
    (
        "manipulate_pipette_two_beaker",
        "configs/other_tasks/manipulate_pipette_two_beaker/action_config.json",
        "robosynchallenge.tasks._other_tasks.manipulate_pipette_two_beaker.action_bank",
        "ManipulatePipetteTwoBeakerActionBank",
    ),
    (
        "pour_water",
        "configs/other_tasks/pour_water/action_config.json",
        "robosynchallenge.tasks._other_tasks.pour_water.action_bank",
        "PourWaterActionBank",
    ),
    (
        "sample_loading-v1",
        "configs/other_tasks/sample_loading/action_config.json",
        "robosynchallenge.tasks._other_tasks.sample_loading.action_bank",
        "SampleLoadingActionBank",
    ),
]


COMMON_EDGE_NAMES = (
    "execute_open",
    "execute_close",
    "plan_trajectory",
    "stand_still",
)

CONFIG_ROOT = Path("configs")
LEGACY_ABSOLUTE_PREFIXES = (
    "/root/workspace/RoboSynChallenge",
    "/root/workspace/outputs_pan",
    "/root/workspace/embodichain",
    "/home/dex/EmbodiChain/RoboSynChallenge",
)
OTHER_TASK_ENV_IDS = (
    "OpenPanPickAndPlaceEnv-v1",
    "ManipulatePipetteTwoBeaker",
)


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _walk_json_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_values(child)
    else:
        yield value


def _collect_action_bank_functions(action_bank_cls: type) -> tuple[dict, dict]:
    node_funcs = {}
    edge_funcs = {}
    action_bank_classes = [
        cls for cls in reversed(action_bank_cls.mro()) if issubclass(cls, ActionBank)
    ]
    for cls in action_bank_classes:
        node_funcs.update(get_func_tag("node").functions.get(cls.__name__, {}))
        edge_funcs.update(get_func_tag("edge").functions.get(cls.__name__, {}))
    return node_funcs, edge_funcs


def test_target_action_configs_load_and_gripper_edges_have_control_part() -> None:
    missing_or_invalid = []
    for label, config_path, _, _ in TASK_ACTION_BANKS:
        action_config = _read_json(config_path)
        assert action_config["scope"]
        assert action_config["node"]
        assert action_config["edge"]

        for scope, edges in action_config["edge"].items():
            for edge in edges:
                edge_label, spec = next(iter(edge.items()))
                if spec.get("name") not in {"execute_open", "execute_close"}:
                    continue
                control_part = spec.get("kwargs", {}).get("control_part")
                expected = scope if scope in {"left_eef", "right_eef"} else None
                if control_part not in {"left_eef", "right_eef"}:
                    missing_or_invalid.append((label, edge_label, control_part))
                if expected is not None and control_part != expected:
                    missing_or_invalid.append((label, edge_label, control_part))

    assert missing_or_invalid == []


def test_all_configs_load_and_do_not_use_legacy_absolute_paths() -> None:
    offenders = []
    for config_path in sorted(CONFIG_ROOT.glob("**/*.json")):
        config = _read_json(str(config_path))
        for value in _walk_json_values(config):
            if not isinstance(value, str):
                continue
            if value.startswith(LEGACY_ABSOLUTE_PREFIXES):
                offenders.append((str(config_path), value))

    assert offenders == []


def test_robosynchallenge_data_namespace_resolves_local_assets() -> None:
    resolved = Path(get_data_path("RoboSynChallenge/assets/button/button.urdf"))

    assert resolved.is_file()
    assert resolved.name == "button.urdf"


def test_embodichain_visual_material_runtime_shim_parses_shape_cfg() -> None:
    from embodichain.lab.sim.shapes import ShapeCfg

    shape_cfg = ShapeCfg.from_dict(
        {
            "shape_type": "Cube",
            "size": [1.0, 1.0, 1.0],
            "visual_material": {
                "uid": "unit_test_material",
                "base_color": [1.0, 1.0, 1.0, 1.0],
            },
        }
    )

    assert shape_cfg.visual_material.uid == "unit_test_material"


def test_other_task_envs_are_registered() -> None:
    import robosynchallenge.tasks  # noqa: F401

    missing = [env_id for env_id in OTHER_TASK_ENV_IDS if env_id not in registry]

    assert missing == []


def test_configured_robosynchallenge_asset_paths_exist() -> None:
    missing = []
    for config_path in sorted(CONFIG_ROOT.glob("**/*.json")):
        config = _read_json(str(config_path))
        for value in _walk_json_values(config):
            if not isinstance(value, str):
                continue
            if not value.startswith("RoboSynChallenge/"):
                continue
            resolved = Path(get_data_path(value))
            if not resolved.exists():
                missing.append((str(config_path), value))

    assert missing == []


def test_action_banks_parse_graphs_with_atomic_replay_common_edges() -> None:
    for label, config_path, module_name, class_name in TASK_ACTION_BANKS:
        module = importlib.import_module(module_name)
        action_bank_cls = getattr(module, class_name)
        action_bank = action_bank_cls(_read_json(config_path))

        node_funcs, edge_funcs = _collect_action_bank_functions(action_bank_cls)
        action_bank.parse_network(node_funcs, edge_funcs, vis_graph=False)

        assert issubclass(action_bank_cls, AtomicEdgeReplayActionBankMixin), label
        for func_name in COMMON_EDGE_NAMES:
            if label == "table_rearrangement" and func_name == "plan_trajectory":
                assert edge_funcs[func_name] is getattr(
                    ExplicitTcpAtomicReplayActionBankMixin, func_name
                ), (label, func_name)
                continue
            assert edge_funcs[func_name] is getattr(
                AtomicEdgeReplayActionBankMixin, func_name
            ), (label, func_name)
        assert node_funcs["execute_open"] is AtomicEdgeReplayActionBankMixin.execute_open
        assert (
            node_funcs["execute_close"]
            is AtomicEdgeReplayActionBankMixin.execute_close
        )


def test_table_rearrangement_uses_explicit_tcp_replay_for_arm_edges() -> None:
    from robosynchallenge.tasks.table_rearrangement.action_bank import (
        TableRearrangementActionBank,
    )

    _, edge_funcs = _collect_action_bank_functions(TableRearrangementActionBank)

    assert issubclass(
        TableRearrangementActionBank,
        ExplicitTcpAtomicReplayActionBankMixin,
    )
    assert (
        edge_funcs["plan_trajectory"]
        is ExplicitTcpAtomicReplayActionBankMixin.plan_trajectory
    )


def test_explicit_tcp_replay_returns_local_arm_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeRobot:
        device = torch.device("cpu")

        def get_qpos(self):
            return torch.zeros(1, 4, dtype=torch.float32)

        def get_joint_ids(self, name):
            assert name == "left_arm"
            return [0, 1]

    class FakeAction:
        arm_joint_ids = [0, 1]

        def __init__(self) -> None:
            self.cfg = SimpleNamespace(sample_interval=0)

        def execute(self, target, state):
            calls.append((self.cfg.sample_interval, target.xpos[0, 0, 3].item()))
            full_trajectory = torch.zeros(
                1, self.cfg.sample_interval, 4, dtype=torch.float32
            )
            full_trajectory[:, :, 0] = target.xpos[0, 0, 3]
            full_trajectory[:, :, 1] = target.xpos[0, 0, 3] + 10.0
            return SimpleNamespace(
                success=True,
                trajectory=full_trajectory,
                next_state=SimpleNamespace(last_qpos=full_trajectory[:, -1, :]),
            )

    def fake_compute_tcp_pose(env, control_part, qpos):
        pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        pose[:, 0, 3] = qpos[0]
        return pose

    def fake_make_action(env, *, control_part, sample_interval):
        assert control_part == "left_arm"
        action = FakeAction()
        action.cfg.sample_interval = sample_interval
        return action

    def fake_make_world_state(env, joint_ids, start_qpos):
        assert joint_ids == [0, 1]
        assert torch.allclose(start_qpos, torch.tensor([0.0, 0.5]))
        return SimpleNamespace(last_qpos=torch.zeros(1, 4, dtype=torch.float32))

    monkeypatch.setattr(
        ExplicitTcpAtomicReplayActionBankMixin,
        "_compute_tcp_pose_from_qpos",
        staticmethod(fake_compute_tcp_pose),
    )
    monkeypatch.setattr(
        ExplicitTcpAtomicReplayActionBankMixin,
        "_make_move_end_effector_action",
        staticmethod(fake_make_action),
    )
    monkeypatch.setattr(
        ExplicitTcpAtomicReplayActionBankMixin,
        "_make_world_state",
        staticmethod(fake_make_world_state),
    )
    monkeypatch.setattr(
        ExplicitTcpAtomicReplayActionBankMixin,
        "_end_effector_pose_target",
        staticmethod(lambda xpos: SimpleNamespace(xpos=xpos)),
    )
    env = SimpleNamespace(
        robot=FakeRobot(),
        affordance_datas={
            "start": torch.tensor([0.0, 0.5]),
            "middle": torch.tensor([1.0, 1.5]),
            "end": torch.tensor([3.0, 3.5]),
        },
    )

    replay = ExplicitTcpAtomicReplayActionBankMixin.plan_trajectory(
        env,
        agent_uid="left_arm",
        keypose_names=["start", "middle", "end"],
        duration=5,
        edge_name="unit_test_edge",
    )

    assert calls == [(3, 1.0), (2, 3.0)]
    assert replay.shape == (2, 5)
    assert np.allclose(replay[0], [1.0, 1.0, 1.0, 3.0, 3.0])
    assert np.allclose(replay[1], [11.0, 11.0, 11.0, 13.0, 13.0])


def test_duration_split_preserves_total_length() -> None:
    assert AtomicEdgeReplayActionBankMixin._split_duration(11, 3) == [4, 4, 3]


def test_qpos_tensor_flattens_single_env_values() -> None:
    qpos = AtomicEdgeReplayActionBankMixin._to_qpos_tensor(
        torch.ones(1, 6),
        device=torch.device("cpu"),
    )

    assert qpos.shape == (6,)
    assert torch.allclose(qpos, torch.ones(6))


def test_match_qpos_dim_broadcasts_scalar_gripper_target() -> None:
    qpos = AtomicEdgeReplayActionBankMixin._match_qpos_dim(torch.tensor([0.05]), 2)

    assert torch.allclose(qpos, torch.tensor([0.05, 0.05]))


def test_gripper_replay_keeps_original_waypoints_and_control_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"sample_intervals": []}
    original_qpos = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    def fake_original_func(env, *, func_name, control_part, duration, **kwargs):
        calls["original"] = (func_name, control_part, duration)
        return original_qpos

    class FakeAction:
        joint_ids = [13]
        n_envs = 1

        def __init__(self) -> None:
            self.cfg = SimpleNamespace(sample_interval=0)

        def execute(self, target, state):
            calls["sample_intervals"].append(self.cfg.sample_interval)
            full_trajectory = torch.zeros(1, 2, 14, dtype=torch.float32)
            full_trajectory[:, -1, self.joint_ids] = target.qpos.reshape(1, 1)
            return SimpleNamespace(
                success=True,
                trajectory=full_trajectory,
                next_state=state,
            )

    def fake_make_action(env, *, control_part, sample_interval):
        calls["action"] = (control_part, sample_interval)
        return FakeAction()

    def fake_make_world_state(env, joint_ids, start_qpos):
        calls["world_state"] = (joint_ids, start_qpos.detach().cpu())
        return object()

    monkeypatch.setattr(
        AtomicEdgeReplayActionBankMixin,
        "_call_original_gripper_func",
        staticmethod(fake_original_func),
    )
    monkeypatch.setattr(
        AtomicEdgeReplayActionBankMixin,
        "_make_move_joints_action",
        staticmethod(fake_make_action),
    )
    monkeypatch.setattr(
        AtomicEdgeReplayActionBankMixin,
        "_make_world_state",
        staticmethod(fake_make_world_state),
    )
    env = SimpleNamespace(robot=SimpleNamespace(device=torch.device("cpu")))

    replay = AtomicEdgeReplayActionBankMixin._execute_gripper_replay(
        env,
        func_name="execute_close",
        return_action=True,
        control_part="right_eef",
        duration=original_qpos.shape[1],
    )

    assert calls["original"] == ("execute_close", "right_eef", original_qpos.shape[1])
    assert calls["action"] == ("right_eef", original_qpos.shape[1])
    assert calls["world_state"][0] == [13]
    assert calls["sample_intervals"] == [2, 2]
    assert np.allclose(replay, original_qpos)
