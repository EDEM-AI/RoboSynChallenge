# -*- coding: utf-8 -*-
"""P001 单测（不需要仿真 / torch / GPU）：apply_cfg 的 dict 变换、readback 形状、
两段驱动切换、判据锁存、资产自洽。

跑法：python3 tests/run_all.py [--official-config <gym_config.json>]
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import math
import pathlib
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np

HERE = pathlib.Path(__file__).resolve()
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG.parent))  # final/

# the derived assets are generated on the fly from the official mesh into a temp dir
os.environ.setdefault("P001_ASSET_DIR", tempfile.mkdtemp(prefix="p001_assets_"))
from envpatch_p001.patches import P001_mixer_switch as P001  # noqa: E402
from envpatch_p001.patches import P001_criteria as C  # noqa: E402

_REPO = pathlib.Path(os.environ.get("ROBOSYN_REPO") or PKG.parents[2])   # official checkout (extras/ lives inside it)
DEFAULT_OFFICIAL = _REPO / "configs" / "mixer_operating" / "random" / "gym_config.json"


# ---------------------------------------------------------------- 假 env

UP_ROT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # 本体系 y → 世界 z（搅拌器 init_rot）


class FakeArt:
    joint_names = ["flip"]

    def __init__(self, theta=0.0, theta_max=math.radians(30.0), lever_id=501):
        self.q = np.array([[theta]], dtype=np.float32)
        self.tgt = np.array([[0.0]], dtype=np.float32)
        self.lim = np.array([[[0.0, theta_max]]], dtype=np.float32)
        self.drive = [0.1, 0.005, 0.5, 5.0]
        self.root = np.eye(4)[None]
        self.root[0, :3, :3] = UP_ROT
        self.root[0, :3, 3] = [0.6, 0.0, 0.9]
        self.lever_id = lever_id
        self.drive_calls = 0
        self.cfg = types.SimpleNamespace(body_scale=[0.3, 0.3, 0.3])

    def get_qpos(self, target=False):
        return self.tgt.copy() if target else self.q.copy()

    def set_qpos(self, qpos, joint_ids=None, env_ids=None, target=True):
        arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if target:
            self.tgt = arr.reshape(1, -1)
        else:
            self.q = arr.reshape(1, -1)

    def get_qpos_limits(self, joint_ids=None, env_ids=None):
        return self.lim.copy()

    def set_qpos_limits(self, qpos_limits, joint_ids=None, env_ids=None):
        self.lim = np.asarray(qpos_limits, dtype=np.float32).reshape(1, 1, 2)

    def get_joint_drive(self, joint_ids=None, env_ids=None):
        return tuple(np.array([[v]], dtype=np.float32) for v in self.drive)

    def set_joint_drive(self, **kw):
        self.drive_calls += 1
        self.drive = [kw["stiffness"], kw["damping"], kw["max_effort"], kw["max_velocity"]]

    def get_local_pose(self, to_matrix=False):
        return self.root.copy()

    def set_local_pose(self, pose, env_ids=None):
        p = np.asarray(pose, dtype=np.float32)
        if p.ndim == 2:
            p = p[None]
        self.root = p.copy()

    def get_user_ids(self, link_name=None):
        return np.array([[self.lever_id]], dtype=np.int32)

    def get_link_pose(self, link_name, env_ids=None, to_matrix=False):
        import math as _m
        th = float(self.q[0, 0])
        c, s = _m.cos(th), _m.sin(th)
        M = np.eye(4)
        M[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        M[:3, 3] = [-0.105, 0.0, 0.038]
        return self.root @ M[None]


class FakeMixer:
    def __init__(self):
        self.root = np.eye(4)[None]
        self.root[0, :3, :3] = UP_ROT
        self.root[0, :3, 3] = [0.6, 0.0, 0.9]
        self.cfg = types.SimpleNamespace(body_scale=[0.3, 0.3, 0.3])

    def get_local_pose(self, to_matrix=False):
        return self.root.copy()

    def get_user_ids(self):
        return np.array([101], dtype=np.int32)


class FakeRobot:
    def get_user_ids(self, link_name=None):
        return {"left_link7": np.array([[301]], dtype=np.int32),
                "left_link8": np.array([[302]], dtype=np.int32)}[link_name]


class FakeSensor:
    def __init__(self):
        self.valid = np.array([[True, False]])
        self.uids = np.array([[[501, 301], [0, 0]]], dtype=np.int32)
        self.imp = np.array([[0.02, 0.0]])

    def get_data(self):
        return {"is_valid": self.valid, "user_ids": self.uids, "impulse": self.imp}


class FakeSim:
    def __init__(self):
        self.art = FakeArt()
        self.mixer = FakeMixer()
        self.sensor = FakeSensor()

    def get_articulation(self, uid):
        return self.art if uid == "mixer_switch" else None

    def get_rigid_object(self, uid):
        return self.mixer if uid == "beaker_mixer" else None

    def get_sensor(self, uid):
        return self.sensor


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.robot = FakeRobot()
        self.control_freq = 25
        self._mixer_user_ids = np.array([101], dtype=np.int32)


class FakeWrapper:
    """模拟 gym.make 的包装环境：只有 .unwrapped 指到底层（P1-1 回归用）。"""

    def __init__(self, base):
        self.unwrapped = base


# ---------------------------------------------------------------- 1 · apply_cfg

def test_apply_cfg(official_cfg):
    src = copy.deepcopy(official_cfg)
    out = P001.apply_cfg(official_cfg)

    assert official_cfg == src, "apply_cfg 不该改入参"

    mixer = next(o for o in out["rigid_object"] if o["uid"] == "beaker_mixer")
    assert mixer["shape"]["fpath"] == str(P001.BODY_OBJ), mixer["shape"]["fpath"]
    assert mixer["body_scale"] == [0.3, 0.3, 0.3]
    assert mixer["max_convex_hull_num"] == 12

    art = next(o for o in out["articulation"] if o["uid"] == "mixer_switch")
    assert art["fpath"] == str(P001.URDF)
    assert art["fix_base"] is True
    assert art["body_scale"] == mixer["body_scale"], "拨杆要和机身一起缩放"
    assert art["init_pos"] == mixer["init_pos"]
    assert art["drive_pros"]["stiffness"] == 0.1
    assert art["drive_pros"]["damping"] == 0.005
    assert art["drive_pros"]["drive_type"] == "force"

    ev = out["env"]["events"]
    keys = list(ev)
    i_mix = keys.index("random_beaker_mixer_pose")
    i_follow = keys.index("p001_follow_mixer_pose")
    assert i_follow == i_mix + 1, f"根位姿事件要排在随机摆位之后：{keys[i_mix:i_follow + 1]}"
    assert ev["p001_follow_mixer_pose"]["mode"] == "reset"
    assert ev["p001_follow_mixer_pose"]["func"] == "p001_follow_root"
    assert ev["p001_switch_toggle"]["mode"] == "interval"
    assert ev["p001_switch_toggle"]["interval_step"] == 1
    assert ev["p001_switch_toggle"]["func"] == "p001_toggle_step"

    attrs = {a["name"]: a for a in ev["prepare_extra_attr"]["params"]["attrs"]}
    for k in ("switch_pivot_x", "switch_pivot_y", "switch_pivot_z", "switch_axis",
              "switch_theta_max", "switch_joint", "switch_lever_link"):
        assert k in attrs, k
    assert abs(attrs["switch_theta_max"]["value"] - math.radians(30)) < 1e-9

    reg = next(e for e in ev["register_info_to_env"]["params"]["registry"]
               if e["entity_cfg"]["uid"] == "beaker_mixer")
    for k in ("switch_pivot_x", "switch_axis", "switch_theta_max"):
        assert k in reg["attrs"]

    cs = next(s for s in out["sensor"] if s["uid"] == "beaker_mixer_button_contact")
    assert any(a["articulation_uid"] == "mixer_switch" for a in cs["articulation_cfg_list"]), \
        "接触传感器要收拨杆"
    lever = next(a for a in cs["articulation_cfg_list"] if a["articulation_uid"] == "mixer_switch")
    assert lever["link_name_list"] == ["switch_lever"]

    # 幂等
    out2 = P001.apply_cfg(out)
    assert len([o for o in out2["articulation"] if o["uid"] == "mixer_switch"]) == 1
    assert len([k for k in out2["env"]["events"] if k.startswith("p001_")]) == 2
    assert len([a for a in next(s for s in out2["sensor"] if s["uid"] == "beaker_mixer_button_contact")
                ["articulation_cfg_list"] if a["articulation_uid"] == "mixer_switch"]) == 1

    # θmax 可改（随机轴）
    out3 = P001.apply_cfg(official_cfg, {"theta_max_deg": 24.0})
    a3 = {a["name"]: a for a in out3["env"]["events"]["prepare_extra_attr"]["params"]["attrs"]}
    assert abs(a3["switch_theta_max"]["value"] - math.radians(24)) < 1e-9
    assert out3["env"]["events"]["p001_switch_toggle"]["params"]["theta_max_deg"] == 24.0

    # 重入换 θmax：功能部位的值要跟着变（P2：不能留旧的 30°）
    out4 = P001.apply_cfg(out3, {"theta_max_deg": 22.0})
    a4 = {a["name"]: a for a in out4["env"]["events"]["prepare_extra_attr"]["params"]["attrs"]}
    assert abs(a4["switch_theta_max"]["value"] - math.radians(22)) < 1e-9, a4["switch_theta_max"]
    assert out4["env"]["events"]["p001_switch_toggle"]["params"]["theta_max_deg"] == 22.0
    assert len([a for a in out4["env"]["events"]["prepare_extra_attr"]["params"]["attrs"]
                if a["name"] == "switch_theta_max"]) == 1

    # 不是搅拌任务就报错，不半改
    other = copy.deepcopy(official_cfg)
    other["rigid_object"] = [o for o in other["rigid_object"] if o["uid"] != "beaker_mixer"]
    try:
        P001.apply_cfg(other)
    except RuntimeError as e:
        assert "beaker_mixer" in str(e)
    else:
        raise AssertionError("没有 beaker_mixer 时应该报错")


# ---------------------------------------------------------------- 2 · readback / verify

def test_readback_and_verify():
    env = FakeEnv()
    env.sim.art.root = env.sim.mixer.root.copy()
    rb = P001.apply_env(env)
    assert rb["patch"] == "P001"
    assert rb["joint"] == "flip"
    assert abs(rb["theta_max_rad"] - math.radians(30)) < 1e-5
    assert rb["theta_rad"] == [0.0]
    assert abs(rb["drive"]["stiffness"] - 0.1) < 1e-6
    assert rb["root_dev_m"] < 1e-6
    assert rb["lever_contact"] is True, "假传感器里拨杆×left_link7 有接触"
    assert abs(rb["lever_contact_impulse"] - 0.02) < 1e-9
    assert rb["mixer_body_scale"] == [0.3, 0.3, 0.3]
    assert rb["contact_ok"] is True
    assert P001.verify(rb) == []

    # 官方判据看得到拨杆
    assert 501 in list(np.asarray(env._mixer_user_ids).reshape(-1))
    assert 101 in list(np.asarray(env._mixer_user_ids).reshape(-1))

    # θmax 读回不对 → 报
    env2 = FakeEnv()
    env2.sim.art.lim = np.array([[[0.0, math.radians(20)]]], dtype=np.float32)
    env2.sim.art.root = env2.sim.mixer.root.copy()
    rb2 = P001.readback(env2)
    bad = P001.verify(rb2)
    assert any("θmax" in b for b in bad), bad

    # 根位姿偏了 → 报
    env3 = FakeEnv()
    env3.sim.art.root = env3.sim.mixer.root.copy()
    env3.sim.art.root[0, :3, 3] += [0.01, 0, 0]
    rb3 = P001.readback(env3)
    assert any("根位姿" in b for b in P001.verify(rb3))

    # 驱动读回不对 → 报
    env4 = FakeEnv()
    env4.sim.art.root = env4.sim.mixer.root.copy()
    env4.sim.art.drive = [0.2, 0.005, 0.5, 5.0]
    rb4 = P001.readback(env4)
    assert any("stiffness" in b for b in P001.verify(rb4))

    # 开关点世界位置偏 1 cm（= body_scale 没作用到关节原点时的样子）→ 报
    env5 = FakeEnv()
    env5.sim.art.root = env5.sim.mixer.root.copy()
    orig = env5.sim.art.get_link_pose
    env5.sim.art.get_link_pose = lambda *a, **k: orig(*a, **k) + np.array([[[0.01, 0, 0]]])
    rb5 = P001.readback(env5)
    assert any("开关点" in b for b in P001.verify(rb5)), P001.verify(rb5)

    # 读不到 link 位姿 → 也报（不许悄悄过）
    env6 = FakeEnv()
    env6.sim.art.root = env6.sim.mixer.root.copy()
    env6.sim.art.get_link_pose = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no link"))
    rb6 = P001.readback(env6)
    assert any("link 位姿" in b for b in P001.verify(rb6))


def test_verify_rejects_bad_readings():
    """P1-2：NaN / 缺项 / 形状不对 / 接触对象读不到，一律不许过。"""
    env = FakeEnv()
    env.sim.art.root = env.sim.mixer.root.copy()
    good = P001.apply_env(env)
    assert P001.verify(good) == []

    for key in ("theta_max_rad", "theta_min_rad", "root_dev_m", "root_dev_deg", "lever_tip_err_m"):
        for bad_val in (float("nan"), float("inf"), None, "x"):
            rb = dict(good)
            rb[key] = bad_val
            assert P001.verify(rb), f"{key}={bad_val!r} 应该被拒"

    for k in ("stiffness", "damping", "max_effort", "max_velocity"):
        rb = dict(good)
        rb["drive"] = dict(good["drive"])
        rb["drive"][k] = float("nan")
        assert any(k in b for b in P001.verify(rb)), k
    rb = dict(good)
    rb["drive"] = None
    assert P001.verify(rb)

    for bs in (None, [], [0.3], [0.3, 0.3], [float("nan")] * 3, ["a", 0.3, 0.3]):
        rb = dict(good)
        rb["mixer_body_scale"] = bs
        assert P001.verify(rb), f"body_scale={bs!r} 应该被拒"
        rb2 = dict(good)
        rb2["art_body_scale"] = bs
        assert P001.verify(rb2), f"art body_scale={bs!r} 应该被拒"

    # 接触读不到 ≠ 没接触
    env7 = FakeEnv()
    env7.sim.art.root = env7.sim.mixer.root.copy()
    env7.robot = None
    env7.sim.get_robot = lambda uid: None
    rb7 = P001.readback(env7)
    assert rb7["contact_ok"] is False and rb7["contact_error"] == "no_robot"
    assert any("接触读不到" in b for b in P001.verify(rb7))

    # 左手 link id 查不到 → no_left_link_ids（P1-2 点名的那条）
    env8 = FakeEnv()
    env8.sim.art.root = env8.sim.mixer.root.copy()
    env8.robot = types.SimpleNamespace(get_user_ids=lambda link_name=None: (_ for _ in ()).throw(KeyError(link_name)))
    rb8 = P001.readback(env8)
    assert rb8["contact_ok"] is False and rb8["contact_error"] == "no_left_link_ids", rb8["contact_error"]
    assert any("接触读不到" in b for b in P001.verify(rb8))

    # 传感器不在 → no_sensor
    env9 = FakeEnv()
    env9.sim.art.root = env9.sim.mixer.root.copy()
    env9.sim.sensor = None
    rb9 = P001.readback(env9)
    assert rb9["contact_error"] == "no_sensor"
    assert any("接触读不到" in b for b in P001.verify(rb9))


def test_wrapper_state_not_split():
    """P1-1：apply_env 拿包装环境、事件和读数在底层环境，状态必须是同一份。"""
    base = FakeEnv()
    base.sim.art.root = base.sim.mixer.root.copy()
    wrapper = FakeWrapper(base)
    rb = P001.apply_env(wrapper)
    assert P001.verify(rb) == []
    assert getattr(base, "_p001_monitor", None) is not None, "监视器必须挂在底层"
    assert getattr(wrapper, "_p001_monitor", None) is None
    # 底层跑事件 → 底层监视器收样本；从包装器读判据也读得到
    base.sim.art.q = np.array([[0.0]], dtype=np.float32)
    P001.toggle_step(base, None)
    assert len(base._p001_monitor.theta) == 1
    m = P001.episode_metrics(wrapper)
    assert m is not None and m["n_steps"] == 1
    # toggle 状态也在底层
    assert getattr(base, "_p001_toggle_state", None) is not None
    assert getattr(wrapper, "_p001_toggle_state", None) is None


# ---------------------------------------------------------------- 3 · 两段驱动

def test_toggle_two_stage():
    env = FakeEnv()
    env.sim.art.root = env.sim.mixer.root.copy()
    P001.apply_env(env)
    art = env.sim.art
    th_max = math.radians(30)
    mid = th_max / 2

    # 关位：target 0
    art.q = np.array([[0.0]], dtype=np.float32)
    P001.toggle_step(env, None, **{k: v for k, v in
                                   dict(theta_max_deg=30.0, hysteresis_deg=2.0).items()})
    assert float(art.tgt[0, 0]) == 0.0

    # 没过中间点：还是 0
    art.q = np.array([[mid - math.radians(5)]], dtype=np.float32)
    P001.toggle_step(env, None)
    assert float(art.tgt[0, 0]) == 0.0

    # 过了中间点 + 滞回：跳到 θmax
    art.q = np.array([[mid + math.radians(3)]], dtype=np.float32)
    P001.toggle_step(env, None)
    assert abs(float(art.tgt[0, 0]) - th_max) < 1e-6

    # 回到滞回带里：保持开位目标（不回弹）
    art.q = np.array([[mid]], dtype=np.float32)
    P001.toggle_step(env, None)
    assert abs(float(art.tgt[0, 0]) - th_max) < 1e-6

    # 退到中间点以下 - 滞回：回关位目标
    art.q = np.array([[mid - math.radians(3)]], dtype=np.float32)
    P001.toggle_step(env, None)
    assert float(art.tgt[0, 0]) == 0.0

    # 监视器在收样本
    mon = env._p001_monitor
    assert len(mon.theta) >= 5
    assert abs(mon.dt - 1 / 25) < 1e-9


# ---------------------------------------------------------------- 4 · 判据

def test_criteria():
    dt = 1 / 25
    th = math.radians(30)
    n = 60

    # 推到 0.9 θmax 并停 1 s、不碰手 → 成功
    theta = [0.0] * 10 + [0.95 * th] * 25 + [0.95 * th] * 25
    contact = [False] * n
    imp = [0.0] * n
    contact[8] = True
    imp[8] = 0.05   # 推的那一下（在过中间点之前）
    m = C.episode_metrics(theta, contact, imp, dt, th)
    assert m["switched_on"] is True
    assert m["flip_by_left"] is True
    assert abs(m["flip_frac_end"] - 0.95) < 1e-6
    assert m["flip_hold_s"] >= 0.5

    # 到开位但手一直按着（到回合末都没松）→ 不算停住
    contact2 = [False] * n
    contact2[10:] = [True] * (n - 10)
    m2 = C.episode_metrics(theta, contact2, [0.05 if c else 0.0 for c in contact2], dt, th)
    assert m2["switched_on"] is False

    # 只停 0.48 s → 不够
    theta3 = [0.0] * 10 + [0.95 * th] * 12 + [0.0] * 20
    m3 = C.episode_metrics(theta3, [False] * len(theta3), [0.0] * len(theta3), dt, th)
    assert m3["switched_on"] is False, m3["flip_hold_s"]

    # 推了但没到 0.8 → 不算
    theta4 = [0.0] * 10 + [0.7 * th] * 30 + [0.0] * 10
    m4 = C.episode_metrics(theta4, [False] * len(theta4), [0.0] * len(theta4), dt, th)
    assert m4["switched_on"] is False

    # 接触在过中间点之后（比如被烧杯碰的）→ 不算左手推的
    theta5 = [0.0] * 10 + [0.95 * th] * 20
    c5 = [False] * len(theta5)
    i5 = [0.0] * len(theta5)
    c5[25] = True
    i5[25] = 0.5
    m5 = C.episode_metrics(theta5, c5, i5, dt, th)
    assert m5["flip_by_left"] is False

    # 冲量不够 → 不算
    m6 = C.episode_metrics(theta, contact, [0.005 if c else 0.0 for c in contact], dt, th)
    assert m6["flip_by_left"] is False

    # 没过中间点 → 没有「推过」这回事
    m7 = C.episode_metrics([0.0] * 5 + [0.3 * th] * 20, [True] * 25, [0.1] * 25, dt, th)
    assert m7["flip_by_left"] is False and m7["switched_on"] is False


# ---------------------------------------------------------------- 5 · 资产自洽

def test_assets():
    import yaml
    P001.ensure_assets()
    for name in ("mixer_switch.urdf", "mixer_switch_lever.obj", "mixer_switch_lever_hull.stl",
                 "beaker_mixer_body.obj", "manifest.yaml", "beaker_mixer.mtl"):
        assert (P001.ASSET_DIR / name).exists(), name

    root = ET.parse(P001.ASSET_DIR / "mixer_switch.urdf").getroot()
    joint = root.find("joint")
    assert joint.get("name") == "flip" and joint.get("type") == "revolute"
    assert joint.find("axis").get("xyz") == "0 1 0"
    lim = joint.find("limit")
    assert abs(float(lim.get("upper")) - math.radians(30)) < 1e-5
    assert abs(float(lim.get("lower"))) < 1e-9
    # URDF 用原始网格系（body_scale 之前）：转轴 = -0.105/0.3, 0, 0.038/0.3
    assert joint.find("origin").get("xyz") == "-0.350000 0.000000 0.126667"

    man = yaml.safe_load((P001.ASSET_DIR / "manifest.yaml").read_text(encoding="utf-8"))
    import hashlib
    for name, meta in man["files"].items():
        h = hashlib.sha256((P001.ASSET_DIR / name).read_bytes()).hexdigest()
        assert h == meta["sha256"], f"{name} sha 变了（manifest 没重生成？）"

    # 拨杆网格在 link 局部系：包围盒和 manifest 对得上
    vs = []
    for line in open(P001.ASSET_DIR / "mixer_switch_lever.obj", encoding="utf-8"):
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
    V = np.array(vs)
    assert len(V) > 0
    lo, hi = V.min(0) * 1000, V.max(0) * 1000
    for a, b in zip(lo, np.array(man["lever"]["hull_bbox_local_mm"])[0]):
        assert abs(a - b) < 0.5, (a, b)
    for a, b in zip(hi, np.array(man["lever"]["hull_bbox_local_mm"])[1]):
        assert abs(a - b) < 0.5, (a, b)

    # 机身网格里不该再有一整块拨杆（切掉 1611 面；源 277555 面）
    n_body = sum(1 for line in open(P001.ASSET_DIR / "beaker_mixer_body.obj", encoding="utf-8")
                 if line.startswith("f "))
    assert n_body == man["body"]["n_faces"], (n_body, man["body"]["n_faces"])
    assert man["source"]["obj_sha256"].startswith("412ea93b"), "official beaker_mixer.obj changed"
    assert man["body"]["n_faces"] + man["lever"]["n_faces"] == 277555


# ---------------------------------------------------------------- 6 · 冒烟通过条件

def test_smoke_checks():
    """P1-3：动态坏态、观测缺失、没拨动，都必须出现在问题清单里（= 非零退出）。"""
    th = math.radians(30)
    dt = 1 / 25
    env = FakeEnv()
    env.sim.art.root = env.sim.mixer.root.copy()
    rb = P001.apply_env(env)
    rb["body_shift_m"] = 0.0
    assert P001.verify(rb) == []

    def mon_good(n=40):
        m = C.SwitchMonitor(dt, th)
        for i in range(n):
            m.update(0.0 if i < 10 else 0.95 * th, i == 8, 0.05 if i == 8 else 0.0)
        return m

    good = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb,
                             max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                             final_readback=True)
    assert good == [], good

    # ① 20 步 θ 恒 0、无接触、没拨动 → 必须报（旧版这里退出 0）
    m0 = C.SwitchMonitor(dt, th)
    for _ in range(20):
        m0.update(0.0, False, 0.0)
    bad0 = P001.smoke_checks(n_steps=20, monitor=m0, rb_final=rb,
                             max_body_speed_mps=0.0, max_lever_tip_speed_mps=0.0,
                             final_readback=True)
    assert any("θ 全程" in b for b in bad0), bad0
    assert any("接触" in b for b in bad0), bad0
    assert any("没停住" in b for b in bad0), bad0
    assert any("不是左手推" in b for b in bad0), bad0

    # 2) body shift is a soft condition: reported, not judged
    rb2 = dict(rb)
    rb2["body_shift_m"] = 0.2
    rb2["body_shift_max_m"] = 0.2
    bad2 = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb2,
                             max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                             final_readback=True)
    assert not any("机身被推走" in b for b in bad2), bad2

    # ②b 硬项：推倒 / 推下台面 / 推出台面 / 姿态读不到
    for key, val, word in (("mixer_tilt_deg", 60.0, "推倒"),
                           ("mixer_z_drop_m", 0.05, "推下台面"),
                           ("mixer_xy_shift_m", 0.6, "推出台面")):
        rbx = dict(rb)
        rbx[key] = val
        badx = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rbx,
                                 max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                                 final_readback=True)
        assert any(word in b for b in badx), (key, badx)
    rby = dict(rb)
    rby["mixer_tilt_deg"] = None
    rby["mixer_tilt_max_deg"] = None
    bady = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rby,
                             max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                             final_readback=True)
    assert any("姿态读不到" in b for b in bady), bady

    # ③ 异常速度 → 必须报
    bad3 = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb,
                             max_body_speed_mps=2.0, max_lever_tip_speed_mps=0.1,
                             final_readback=True)
    assert any("机身速度" in b for b in bad3), bad3
    bad3b = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb,
                              max_body_speed_mps=0.1, max_lever_tip_speed_mps=2.0,
                              final_readback=True)
    assert any("拨杆外端速度" in b for b in bad3b), bad3b

    # ③c 首步飞出：锁存在 readback 里的最大速度（调用方没另传）也要算
    rb3 = dict(rb)
    rb3["body_speed_max_mps"] = 5.0
    bad3c = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb3,
                              max_body_speed_mps=0.0, max_lever_tip_speed_mps=0.0,
                              final_readback=True)
    assert any("机身速度" in b for b in bad3c), bad3c

    # ④ 观测缺失：样本数对不上 / 接触读不到 / 监视器没挂
    bad4 = P001.smoke_checks(n_steps=41, monitor=mon_good(), rb_final=rb,
                             max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                             final_readback=True)
    assert any("观测缺失" in b for b in bad4), bad4
    m5 = mon_good()
    m5.contact_error_steps = 3
    m5.last_contact_error = "no_left_link_ids"
    bad5 = P001.smoke_checks(n_steps=40, monitor=m5, rb_final=rb,
                             max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                             final_readback=True)
    assert any("接触有" in b for b in bad5), bad5
    assert P001.smoke_checks(n_steps=40, monitor=None, rb_final=rb,
                             max_body_speed_mps=0.0, max_lever_tip_speed_mps=0.0)

    # ④b 中途一次坏观测（NaN 位姿 / θ NaN）→ 必须报，不能靠恢复洗掉
    rb4b = dict(rb)
    rb4b["pose_invalid_steps"] = 1
    rb4b["dynamics_error"] = "pose_nonfinite"
    bad4b = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb4b,
                              max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                              final_readback=True)
    assert any("机身位姿无效" in b for b in bad4b), bad4b
    rb4c = dict(rb)
    rb4c["theta_invalid_steps"] = 2
    bad4c = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb4c,
                              max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                              final_readback=True)
    assert any("θ 无效" in b for b in bad4c), bad4c
    bad4d = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb,
                              max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                              pose_invalid_steps=1, final_readback=True)
    assert any("机身位姿无效" in b for b in bad4d), bad4d

    # ⑤ 末态读回坏（NaN / body_shift 读不到）→ 必须报
    rb6 = dict(rb)
    rb6["lever_tip_err_m"] = float("nan")
    bad6 = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb6,
                             max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                             final_readback=True)
    assert any("开关点" in b for b in bad6), bad6
    rb7 = dict(rb)
    rb7["body_shift_m"] = None
    rb7.pop("body_shift_max_m", None)
    bad7 = P001.smoke_checks(n_steps=40, monitor=mon_good(), rb_final=rb7,
                             max_body_speed_mps=0.1, max_lever_tip_speed_mps=0.1,
                             final_readback=True)
    assert any("机身挪动读不到" in b for b in bad7), bad7


def test_latch_dynamics():
    """P1-3 残余：中途偏移后归位、首步跳变，都要被锁存（不能只看末态/第二步以后）。"""
    env = FakeEnv()
    env.sim.art.root = env.sim.mixer.root.copy()
    P001.apply_env(env)
    mixer = env.sim.mixer
    dt = 1 / 25
    mixer.root[0, :3, 3] += [0.005, 0, 0]
    P001.toggle_step(env, None)
    mixer.root[0, :3, 3] -= [0.005, 0, 0]   # 归位
    P001.toggle_step(env, None)
    rb = P001.readback(env)
    assert rb["body_shift_m"] < 1e-9, "末态确实归位了"
    assert abs(rb["body_shift_max_m"] - 0.005) < 1e-6, rb["body_shift_max_m"]
    assert abs(rb["body_speed_max_mps"] - 0.005 / dt) < 1e-6, rb["body_speed_max_mps"]

    # 首步跳 0.2 m：apply_env 之后的第一次 toggle 就要算出来
    env2 = FakeEnv()
    env2.sim.art.root = env2.sim.mixer.root.copy()
    P001.apply_env(env2)
    env2.sim.mixer.root[0, :3, 3] += [0.2, 0, 0]
    P001.toggle_step(env2, None)
    rb2 = P001.readback(env2)
    assert abs(rb2["body_shift_max_m"] - 0.2) < 1e-6
    assert abs(rb2["body_speed_max_mps"] - 0.2 / dt) < 1e-5, rb2["body_speed_max_mps"]

    # 中途被推倒 60° 又回正：倾角锁存（末态 0，最大 60）——硬项看最大值
    env7 = FakeEnv()
    env7.sim.art.root = env7.sim.mixer.root.copy()
    P001.apply_env(env7)
    R60 = np.array([[1, 0, 0], [0, math.cos(math.radians(60)), -math.sin(math.radians(60))],
                    [0, math.sin(math.radians(60)), math.cos(math.radians(60))]])
    env7.sim.mixer.root[0, :3, :3] = R60 @ env7.sim.mixer.root[0, :3, :3]
    P001.toggle_step(env7, None)
    env7.sim.mixer.root[0, :3, :3] = UP_ROT
    P001.toggle_step(env7, None)
    rb7 = P001.readback(env7)
    assert rb7["mixer_tilt_deg"] < 1.0, rb7["mixer_tilt_deg"]
    assert abs(rb7["mixer_tilt_max_deg"] - 60.0) < 0.5, rb7["mixer_tilt_max_deg"]

    # 翻过 90° 要单调：135° → 135°、180° 翻转 → 180°（不能算成 45° / 0°）
    for deg in (135.0, 180.0):
        env8 = FakeEnv()
        env8.sim.art.root = env8.sim.mixer.root.copy()
        P001.apply_env(env8)
        th = math.radians(deg)
        Rx = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
        env8.sim.mixer.root[0, :3, :3] = Rx @ UP_ROT
        P001.toggle_step(env8, None)
        rb8 = P001.readback(env8)
        assert abs(rb8["mixer_tilt_deg"] - deg) < 0.5, (deg, rb8["mixer_tilt_deg"])
        assert abs(rb8["mixer_tilt_max_deg"] - deg) < 0.5, (deg, rb8["mixer_tilt_max_deg"])

    # 拨杆外端速度也锁存：一步转 10° 相当于 0.0219 m × 0.1745 / dt
    env3 = FakeEnv()
    env3.sim.art.root = env3.sim.mixer.root.copy()
    P001.apply_env(env3)
    env3.sim.art.q = np.array([[math.radians(10)]], dtype=np.float32)
    P001.toggle_step(env3, None)
    rb3 = P001.readback(env3)
    want = float(np.linalg.norm(P001.TIP_LINK_M)) * math.radians(10) / dt
    assert abs(rb3["lever_tip_speed_max_mps"] - want) < 1e-5, (rb3["lever_tip_speed_max_mps"], want)

    # 中途一次机身位姿 NaN、随后恢复：坏观测必须锁存（不能被后面的正常值洗掉）
    env4 = FakeEnv()
    env4.sim.art.root = env4.sim.mixer.root.copy()
    P001.apply_env(env4)
    env4.sim.mixer.root[0, 0, 3] = np.nan
    P001.toggle_step(env4, None)
    env4.sim.mixer.root[0, 0, 3] = 0.6   # 恢复
    P001.toggle_step(env4, None)
    rb4 = P001.readback(env4)
    assert rb4["pose_invalid_steps"] == 1, rb4["pose_invalid_steps"]
    assert rb4["dynamics_error"].startswith("pose"), rb4["dynamics_error"]
    assert np.isfinite(rb4["body_shift_max_m"])

    # 读位姿抛错也锁存（不是直接 return 了事）
    env5 = FakeEnv()
    env5.sim.art.root = env5.sim.mixer.root.copy()
    P001.apply_env(env5)
    orig_pose = env5.sim.mixer.get_local_pose
    env5.sim.mixer.get_local_pose = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sim gone"))
    P001.toggle_step(env5, None)
    env5.sim.mixer.get_local_pose = orig_pose
    P001.toggle_step(env5, None)
    rb5 = P001.readback(env5)
    assert rb5["pose_invalid_steps"] == 1, rb5["pose_invalid_steps"]
    assert rb5["dynamics_error"].startswith("pose_read"), rb5["dynamics_error"]

    # θ NaN 也锁存
    env6 = FakeEnv()
    env6.sim.art.root = env6.sim.mixer.root.copy()
    P001.apply_env(env6)
    env6.sim.art.q = np.array([[float("nan")]], dtype=np.float32)
    P001.toggle_step(env6, None)
    rb6 = P001.readback(env6)
    assert rb6["theta_invalid_steps"] == 1, rb6["theta_invalid_steps"]


def test_event_funcs_injected():
    """官方按属性名解析事件（不是 "模块:函数"），所以要注入到 robosynchallenge.managers.events。"""
    import sys
    import types

    mod = types.ModuleType("robosynchallenge.managers.events")
    pkg = types.ModuleType("robosynchallenge")
    pkg.__path__ = []
    man = types.ModuleType("robosynchallenge.managers")
    man.__path__ = []
    man.events = mod
    pkg.managers = man
    sys.modules.update({"robosynchallenge": pkg, "robosynchallenge.managers": man,
                        "robosynchallenge.managers.events": mod})
    try:
        assert P001.install_event_funcs() is True
        assert getattr(mod, "p001_follow_root") is P001.follow_root
        assert getattr(mod, "p001_toggle_step") is P001.toggle_step
        cfg = json.loads(DEFAULT_OFFICIAL.read_text(encoding="utf-8"))
        out = P001.apply_cfg(cfg)
        for name in ("p001_follow_mixer_pose", "p001_switch_toggle"):
            fn = out["env"]["events"][name]["func"]
            assert ":" not in fn, fn
            assert callable(getattr(mod, fn))
    finally:
        for k in ("robosynchallenge", "robosynchallenge.managers",
                  "robosynchallenge.managers.events"):
            sys.modules.pop(k, None)

    # 模块不在时：非 strict 返回 False，strict 报错
    assert P001.install_event_funcs() is False
    try:
        P001.install_event_funcs(strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict 应该报错")


def test_parts():
    """ctx.parts 两件：未缩放值 ×0.3 = 官方 button_offset 那套；θmax 跟着参数走；深拷贝。"""
    P = P001.parts()
    flip = P["mixer_switch"]["flip"]
    push = P["mixer_switch"]["push_point"]
    assert flip["kind"] == "joint" and flip["q_name"] == "flip" and flip["link"] == "switch_lever"
    assert abs(flip["range"][1] - math.radians(30)) < 1e-9 and abs(flip["travel"] - math.radians(30)) < 1e-9
    s = 0.3
    for got, want in zip([v * s for v in flip["pivot"]], [-0.105, 0.0, 0.038]):
        assert abs(got - want) < 1e-6, (got, want)
    for got, want in zip([v * s for v in push["p"]], [-0.1176, 0.0, 0.056]):
        assert abs(got - want) < 1e-6, (got, want)
    assert abs(push["lever"] * s - 0.022) < 5e-4
    assert push["kind"] == "axis" and push["normal"] == [0.0, 0.0, 1.0]

    P2 = P001.parts({"theta_max_deg": 24.0})
    assert abs(P2["mixer_switch"]["flip"]["range"][1] - math.radians(24)) < 1e-9
    P2["mixer_switch"]["flip"]["range"][1] = 9.0
    assert abs(P001.parts()["mixer_switch"]["flip"]["range"][1] - math.radians(30)) < 1e-9


def test_parts_scaled():
    """provider → 当集几何 → 消费点：参考变换和 push_point_at 的数值。"""
    P = P001.parts()
    s3 = [0.3, 0.3, 0.3]
    sc = P001.parts_scaled(P, s3)["mixer_switch"]
    for got, want in zip(sc["flip"]["pivot"], [-0.105, 0.0, 0.038]):
        assert abs(got - want) < 1e-6, (got, want)
    for got, want in zip(sc["push_point"]["p"], [-0.1176, 0.0, 0.056]):
        assert abs(got - want) < 1e-6, (got, want)
    assert abs(sc["push_point"]["lever"] - 0.022) < 5e-4
    assert abs(sc["push_point"]["size"]["length"] - 0.022) < 5e-4
    assert sc["push_point"]["normal"] == [0.0, 0.0, 1.0]

    # 逐轴不等缩放：p.z 按 s.z 变；y 轴仍是 (0,1,0)
    sc2 = P001.parts_scaled(P, [0.3, 0.3, 0.6])["mixer_switch"]
    assert sc2["flip"]["axis"] == [0.0, 1.0, 0.0]
    assert abs(sc2["push_point"]["p"][2] - 0.112) < 1e-6

    # P1 回归：倾斜轴 + 非均匀 s，长度量只能乘一次 |diag(s)·axis|
    syn = {"x": {"a": {"kind": "axis", "p": [0.0, 0.0, 0.0],
                       "axis": [1 / math.sqrt(2), 1 / math.sqrt(2), 0.0],
                       "normal": [1.0, 0.0, 1.0],
                       "size": {"length": 2.0}, "lever": 2.0}}}
    out = P001.parts_scaled(syn, [2.0, 4.0, 1.0])["x"]["a"]
    assert abs(out["size"]["length"] - 2.0 * math.sqrt(10)) < 1e-6, out["size"]["length"]
    assert abs(out["lever"] - 2.0 * math.sqrt(10)) < 1e-6, out["lever"]
    assert abs(out["axis"][0] - 0.4472136) < 1e-6 and abs(out["axis"][1] - 0.8944272) < 1e-6
    # normal 用 diag(1/s) 再归一：(1,0,1)/s=(0.5,0,1) → (0.4472,0,0.8944)
    assert abs(out["normal"][0] - 0.4472136) < 1e-6 and abs(out["normal"][2] - 0.8944272) < 1e-6

    # s 必须是 3 维；非单位 axis 报错（长度口径没定义）
    for bad_s in (0.3, [0.3, 0.3]):
        try:
            P001.parts_scaled(P, bad_s)
        except ValueError:
            pass
        else:
            raise AssertionError(f"s={bad_s!r} 应该报错")
    bad_axis = {"x": {"a": {"axis": [0.0, 2.0, 0.0]}}}
    try:
        P001.parts_scaled(bad_axis, s3)
    except ValueError:
        pass
    else:
        raise AssertionError("非单位 axis 应该报错")

    # θ=0 和 θ=30° 的开关点/法向：点用硬编码期望值（不和生产公式同源）
    p0 = P001.push_point_at(P, s3, 0.0)
    assert all(abs(a - b) < 1e-6 for a, b in zip(p0["p"], [-0.1176, 0.0, 0.056]))
    p30 = P001.push_point_at(P, s3, math.radians(30))
    for got, want in zip(p30["p"], [-0.1069119, 0.0, 0.0598885]):
        assert abs(got - want) < 1e-6, (got, want)
    assert abs(p30["normal"][0] - 0.5) < 1e-6 and abs(p30["normal"][2] - math.cos(math.radians(30))) < 1e-6


def test_parts_scaled_axis_lengths():
    """P1 独立回归：固定解析值覆盖轴对齐/斜轴、均匀/非均匀缩放。"""
    cases = [
        ([1.0, 0.0, 0.0], [2.0, 4.0, 1.0], 4.0),
        ([0.0, -1.0, 0.0], [2.0, 4.0, 1.0], 8.0),
        ([1 / math.sqrt(2), 1 / math.sqrt(2), 0.0], [3.0] * 3, 6.0),
        # counter-example: 原长2，缩后轴向量=(sqrt(2),2sqrt(2),0)，长度2sqrt(10)。
        ([1 / math.sqrt(2), 1 / math.sqrt(2), 0.0], [2.0, 4.0, 1.0], 2 * math.sqrt(10)),
        # 三维斜轴缩成(2,2,2)，原长2，故长度4sqrt(3)。
        ([2 / 3, 1 / 3, 2 / 3], [3.0, 6.0, 3.0], 4 * math.sqrt(3)),
    ]
    for axis, scale, expected in cases:
        raw = {"x": {"a": {"axis": axis, "size": {"length": 2.0}, "lever": 2.0}}}
        before = copy.deepcopy(raw)
        got = P001.parts_scaled(raw, scale)["x"]["a"]
        assert math.isclose(got["size"]["length"], expected, abs_tol=1e-12), (axis, scale, got, expected)
        assert math.isclose(got["lever"], expected, abs_tol=1e-12), (axis, scale, got, expected)
        assert math.isclose(np.linalg.norm(got["axis"]), 1.0, abs_tol=1e-12)
        assert raw == before, "变换不得改写 provider 原始表"
        got["size"]["length"] = -1.0
        assert raw == before, "嵌套 size 必须深拷贝"


def test_parts_scaled_normal_and_points():
    """点按s、法向按1/s；斜切向与法向正交关系须保留。"""
    raw = {"x": {"a": {"p": [1.0, -2.0, 3.0], "pivot": [-1.0, 2.0, -3.0],
                       "axis": [1 / math.sqrt(2), 1 / math.sqrt(2), 0.0],
                       "normal": [1 / math.sqrt(3), -1 / math.sqrt(3), 1 / math.sqrt(3)]}}}
    got = P001.parts_scaled(raw, [2.0, 4.0, 1.0])["x"]["a"]
    np.testing.assert_allclose(got["p"], [2.0, -8.0, 3.0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(got["pivot"], [-2.0, 8.0, -3.0], atol=1e-12, rtol=0)
    # 逆缩放法向与(2,-1,4)同向，长度sqrt(21)，与缩后切向正交。
    np.testing.assert_allclose(got["normal"], np.array([2.0, -1.0, 4.0]) / math.sqrt(21), atol=1e-12, rtol=0)
    assert abs(np.dot(got["axis"], got["normal"])) < 1e-12


def test_push_point_at_oblique_axis():
    """合成关节的90°基向量反例；不以生产Rodrigues副本作oracle，不改变生产角度范围。"""
    axis = [1 / math.sqrt(2), 1 / math.sqrt(2), 0.0]
    raw = {"mixer_switch": {
        "flip": {"pivot": [1.0, 2.0, 3.0], "axis": axis,
                 "range": [0.0, math.pi / 2], "unit": "rad"},
        "push_point": {"p": [1.0, 2.0, 5.0], "axis": axis,
                       "normal": [0.0, 0.0, 1.0], "lever": 2.0}}}
    before = copy.deepcopy(raw)
    # 缩后pivot=(2,8,3)，轴=(1,2,0)/sqrt(5)，起始半径沿+z，长2。
    # 右手转90°后半径沿(2,-1,0)，终点由这两个垂直基向量直接确定。
    p0 = P001.push_point_at(raw, [2.0, 4.0, 1.0], 0.0)
    got = P001.push_point_at(raw, [2.0, 4.0, 1.0], math.pi / 2)
    np.testing.assert_allclose(p0["p"], [2.0, 8.0, 5.0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(got["p"], [2 + 4 / math.sqrt(5), 8 - 2 / math.sqrt(5), 3.0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(got["normal"], [2 / math.sqrt(5), -1 / math.sqrt(5), 0.0], atol=1e-12, rtol=0)
    assert math.isclose(got["lever"], 2 * math.sqrt(10), abs_tol=1e-12)
    assert got["theta"] == math.pi / 2 and raw == before


# ---------------------------------------------------------------- run

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg_path = DEFAULT_OFFICIAL
    if "--official-config" in argv:
        cfg_path = pathlib.Path(argv[argv.index("--official-config") + 1])
    official = json.loads(cfg_path.read_text(encoding="utf-8"))
    tests = [
        ("apply_cfg", lambda: test_apply_cfg(official)),
        ("readback/verify", test_readback_and_verify),
        ("verify_rejects_bad", test_verify_rejects_bad_readings),
        ("wrapper_state", test_wrapper_state_not_split),
        ("toggle_two_stage", test_toggle_two_stage),
        ("criteria", test_criteria),
        ("smoke_checks", test_smoke_checks),
        ("latch_dynamics", test_latch_dynamics),
        ("event_funcs", test_event_funcs_injected),
        ("parts", test_parts),
        ("parts_scaled", test_parts_scaled),
        ("parts_scaled_axis_lengths", test_parts_scaled_axis_lengths),
        ("parts_scaled_normal_and_points", test_parts_scaled_normal_and_points),
        ("push_point_at_oblique_axis", test_push_point_at_oblique_axis),
        ("assets", test_assets),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
        else:
            print(f"[ok]   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} 过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
