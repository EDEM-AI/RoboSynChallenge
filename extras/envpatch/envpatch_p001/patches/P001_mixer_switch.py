# -*- coding: utf-8 -*-
"""P001 - mixer switch as a two-position toggle (body stays rigid, lever becomes its own articulation).

Design notes
------------
- The official repository is never modified: only the in-memory gym config copy and the env instance are patched.
- The body is replaced by a mesh without the lever; the lever is a separate revolute articulation
  (axis = body y, travel 0 -> theta_max, pushing into the body is positive).
- Two-stage drive: every control step the drive target is switched from the current theta
  (hysteresis 2 degrees), so the lever stays where it was pushed (no spring-back).

Interface
---------
    apply_cfg(gym_cfg, params=None) -> gym_cfg   pure dict transform (deep copy)
    apply_env(env, params=None)     -> readback  after reset: poses, drive, targets
    readback(env, params=None)      -> dict      effective values, read back
    verify(readback, params=None)   -> list      list of problems (empty = ok)

Criterion readers (the criterion itself is not decided here):
    toggle_step / follow_root   two event functions (interval / reset), hooked into the config by apply_cfg
    episode_metrics(env)        theta / contact readings for one episode, see P001_criteria.py

Assets: the derived meshes (body without lever, lever, hull, URDF) are generated on the spot from the official
`assets/beaker_mixer/beaker_mixer.obj` by `ensure_assets()` (needs trimesh). They are not shipped.
"""
from __future__ import annotations

import copy
import os
import importlib
import math
import pathlib
import sys

import numpy as np

PATCH_ID = "P001"
PATCH_VERSION = "0.1.16"
PATCH_DIR = pathlib.Path(__file__).resolve().parents[1]
ASSET_DIR = pathlib.Path(os.environ.get("P001_ASSET_DIR") or (PATCH_DIR / "assets" / "mixer_switch"))
BODY_OBJ = ASSET_DIR / "beaker_mixer_body.obj"
URDF = ASSET_DIR / "mixer_switch.urdf"
MANIFEST = ASSET_DIR / "manifest.yaml"

# 方案第三节的数（带依据）；随机轴（θmax 20–35°、刚度 0.05–0.2、阻尼 0.002–0.01）由范围文件给，
# 这里只给缺省值和合法区间。
DEFAULTS = {
    "uid": "mixer_switch",
    "mixer_uid": "beaker_mixer",
    "joint": "flip",
    "lever_link": "switch_lever",
    "base_link": "switch_base",
    "theta_max_deg": 30.0,
    "theta_max_range_deg": [20.0, 35.0],
    "hysteresis_deg": 2.0,
    "stiffness": 0.1,
    "damping": 0.005,
    "max_effort": 0.5,
    "max_velocity": 5.0,
    "left_links": ["left_link7", "left_link8"],
    "contact_sensor": "beaker_mixer_button_contact",
    "impulse_min": 0.01,
    "hold_s": 0.5,
    "on_frac": 0.8,
    "friction": [1.0, 0.8],          # static / dynamic；assumed value; tune against push feel on a GPU run
    "monitor": True,
    "official_judge_sees_lever": True,
    "require_body_obj": True,
}

DRIVE_KEYS = ("stiffness", "damping", "max_effort", "max_velocity")

# 开关点（官方 affordance button_offset，本体系已乘 body_scale）与转轴，都写在这里当几何常量；
# 专家侧应改读 ctx.parts['mixer_switch']['push_point']（字段表第 3 版），这里是补丁自检用。
TIP_SCALED = np.array([-0.1176, 0.0, 0.056])
PIVOT_SCALED = np.array([-0.105, 0.0, 0.038])
BODY_SCALE = 0.3
TIP_LINK_M = TIP_SCALED - PIVOT_SCALED   # 开关点在 switch_lever 系里的位置（缩放后的米制）
LEVER_ARM_M = float(np.linalg.norm(TIP_LINK_M))  # 转轴到开关点，≈ 22 mm


def params(p=None) -> dict:
    out = dict(DEFAULTS)
    if p:
        unknown = sorted(set(p) - set(DEFAULTS))
        if unknown:
            raise ValueError(f"P001 不认识的参数：{unknown}（只认 {sorted(DEFAULTS)}）")
        out.update(p)
    return out


# ---------------------------------------------------------------- helpers

def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _u(env):
    return getattr(env, "unwrapped", env)


def _art(env, p):
    u = _u(env)
    art = u.sim.get_articulation(p["uid"])
    if art is None:
        raise RuntimeError(f"⛔ P001：env 里没有关节体 {p['uid']!r}（apply_cfg 没生效？）")
    return art


def _mixer(env, p):
    u = _u(env)
    obj = u.sim.get_rigid_object(p["mixer_uid"])
    if obj is None:
        raise RuntimeError(f"⛔ P001：env 里没有刚体 {p['mixer_uid']!r}")
    return obj


def _joint_id(art, name):
    names = list(art.joint_names)
    if name not in names:
        raise RuntimeError(f"⛔ P001：关节 {name!r} 不在 {names}")
    return names.index(name)


def _pose_arg(pose):
    """set_local_pose needs a torch tensor in the simulator; numpy is fine in unit tests."""
    arr = np.asarray(pose, dtype=np.float32)
    try:
        import torch
    except ImportError:
        return arr
    return torch.as_tensor(arr)


def _rot_angle_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _pose_dev(A, B):
    """A、B: (N,4,4)。返回 (最大平移差 m, 最大转角差 deg)。"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    dt = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1)
    da = np.asarray([_rot_angle_deg(A[i, :3, :3], B[i, :3, :3]) for i in range(len(A))])
    return float(dt.max()) if len(dt) else 0.0, float(da.max()) if len(da) else 0.0


def _lever_tip_err(env, p, art, theta, mixer_pose=None):
    """开关点世界位置自检：关节体（URDF 原始系 + body_scale）算出来的点，和
    「关节体根位姿 ⊗ (转轴 + Ry(θ)·(开关点−转轴))」比。这一步核 body_scale 有没有作用到关节原点。

    参考系用**关节体根位姿**而不是搅拌器当前位姿：机身被推走时根不动（做法 1 的已知代价），
    那条由 body_shift 单独判；混进来会把「机身挪 4.8 mm」误报成「URDF 对不上」。

    返回 (最大误差 m, 期望点世界坐标 list) 或 (None, None)（读不到 link 位姿时）。
    """
    try:
        link = _np(art.get_link_pose(p["lever_link"], to_matrix=True))
    except Exception:  # noqa: BLE001
        return None, None
    th = np.asarray(theta, dtype=np.float64)
    rel = TIP_SCALED - PIVOT_SCALED
    exp_local = np.empty((len(th), 3))
    for i, t in enumerate(th):
        c, s = math.cos(t), math.sin(t)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        exp_local[i] = PIVOT_SCALED + R @ rel
    try:
        mp = _np(art.get_local_pose(to_matrix=True)).astype(np.float64)
    except Exception:  # noqa: BLE001
        if mixer_pose is None:
            return None, None
        mp = np.asarray(mixer_pose, dtype=np.float64)
    exp_world = np.einsum("nij,nj->ni", mp[:, :3, :3], exp_local) + mp[:, :3, 3]
    act_world = np.einsum("nij,nj->ni", link[:, :3, :3], np.tile(TIP_LINK_M, (len(link), 1))) + link[:, :3, 3]
    err = float(np.linalg.norm(exp_world - act_world, axis=1).max()) if len(link) else None
    return err, exp_world.round(6).tolist()


def _ensure_importable():
    """包目录在 sys.path 上（事件函数注入时也用得到）。"""
    root = str(PATCH_DIR.parent)  # final/
    if root not in sys.path:
        sys.path.insert(0, root)


# 官方 config_to_cfg 不是按 "模块:函数" 解析事件，而是在固定模块表里按属性名找
# （embodichain.lab.gym.utils.gym_utils.config_to_cfg → find_function_from_modules）。
# 所以把两个事件函数注入 robosynchallenge.managers.events 的模块命名空间，配置里只写函数名。
EVENT_MODULE = "robosynchallenge.managers.events"
EVENT_FOLLOW = "p001_follow_root"
EVENT_TOGGLE = "p001_toggle_step"


def install_event_funcs(strict: bool = False) -> bool:
    """把 follow_root / toggle_step 挂到官方事件模块上（只改进程里的模块对象，不动文件）。"""
    try:
        mod = importlib.import_module(EVENT_MODULE)
    except Exception as ex:  # noqa: BLE001
        if strict:
            raise RuntimeError(f"⛔ P001：import {EVENT_MODULE} 失败（{type(ex).__name__}: {ex}）；"
                               f"事件函数没法注入，环境会建不起来") from ex
        return False
    setattr(mod, EVENT_FOLLOW, follow_root)
    setattr(mod, EVENT_TOGGLE, toggle_step)
    return True


# ---------------------------------------------------------------- apply_cfg

def _insert_after(d: dict, after_key: str, key: str, value):
    if key in d:
        d[key] = value
        return
    items = list(d.items())
    out = {}
    placed = False
    for k, v in items:
        out[k] = v
        if k == after_key:
            out[key] = value
            placed = True
    if not placed:
        out[key] = value
    d.clear()
    d.update(out)


def official_mixer_obj(path=None):
    """Locate the official beaker_mixer.obj: argument, $P001_OFFICIAL_OBJ, $ROBOSYN_REPO, or the enclosing checkout."""
    cands = [path, os.environ.get("P001_OFFICIAL_OBJ")]
    if os.environ.get("ROBOSYN_REPO"):
        cands.append(pathlib.Path(os.environ["ROBOSYN_REPO"]) / "assets" / "beaker_mixer" / "beaker_mixer.obj")
    cands.append(PATCH_DIR.parents[2] / "assets" / "beaker_mixer" / "beaker_mixer.obj")   # extras/envpatch/envpatch_p001 -> repo root
    for c in cands:
        if c and pathlib.Path(c).is_file():
            return pathlib.Path(c)
    raise RuntimeError("P001: need the official assets/beaker_mixer/beaker_mixer.obj "
                       "(pass it, or set P001_OFFICIAL_OBJ / ROBOSYN_REPO)")


def ensure_assets(official_obj=None, force=False):
    """Split the official mesh on the spot if the derived assets are not there yet (needs trimesh)."""
    if not force and BODY_OBJ.exists() and URDF.exists() and MANIFEST.exists():
        return ASSET_DIR
    import importlib.util
    src = official_mixer_obj(official_obj)
    spec = importlib.util.spec_from_file_location("p001_split_beaker_mixer",
                                                  PATCH_DIR / "generators" / "split_beaker_mixer.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.main(["--src", str(src), "--out", str(ASSET_DIR)])
    return ASSET_DIR


def apply_cfg(gym_cfg: dict, p: dict | None = None) -> dict:
    """把官方 gym 配置改成补丁版（深拷贝，不动入参）。任何一步对不上就报错，不半改。"""
    p = params(p)
    _ensure_importable()
    install_event_funcs()   # 配置里的事件函数按属性名解析，先注入（import 不到就由调用方 strict 报）
    cfg = copy.deepcopy(gym_cfg)

    if p["require_body_obj"]:
        ensure_assets()
    if not URDF.exists():
        raise RuntimeError(f"P001: missing {URDF} (run ensure_assets() / generators/split_beaker_mixer.py)")

    rigids = cfg.get("rigid_object") or []
    mixer = next((o for o in rigids if o.get("uid") == p["mixer_uid"]), None)
    if mixer is None:
        raise RuntimeError(f"⛔ P001：配置里没有 {p['mixer_uid']!r}（这条补丁只对搅拌任务生效）")

    # 1) 机身换成去掉拨杆的网格；body_scale / 质量 / 凸分解块数照官方
    mixer.setdefault("shape", {})["fpath"] = str(BODY_OBJ)

    # 2) 关节体 mixer_switch（根位姿每集跟搅拌器走，根固定）
    arts = cfg.setdefault("articulation", [])
    art_cfg = next((o for o in arts if o.get("uid") == p["uid"]), None)
    if art_cfg is None:
        art_cfg = {
            "uid": p["uid"],
            "fpath": str(URDF),
            "fix_base": True,
            "disable_self_collision": True,
            "init_qpos": [0.0],
            "min_position_iters": 32,
            "min_velocity_iters": 8,
            "attrs": {
                "static_friction": float(p["friction"][0]),
                "dynamic_friction": float(p["friction"][1]),
                "restitution": 0.0,
            },
            "drive_pros": {
                "drive_type": "force",
                "stiffness": float(p["stiffness"]),
                "damping": float(p["damping"]),
                "max_effort": float(p["max_effort"]),
                "max_velocity": float(p["max_velocity"]),
            },
        }
        arts.append(art_cfg)
    else:  # 幂等：参数变了就改，不重复加
        art_cfg["fpath"] = str(URDF)
        art_cfg.setdefault("drive_pros", {}).update({
            "drive_type": "force",
            "stiffness": float(p["stiffness"]),
            "damping": float(p["damping"]),
            "max_effort": float(p["max_effort"]),
            "max_velocity": float(p["max_velocity"]),
        })
    # 根位姿跟搅拌器一致（体坐标原点相同；body_scale 也照抄，保证拨杆和机身一起缩放）
    art_cfg["init_pos"] = list(mixer.get("init_pos") or [0.0, 0.0, 0.0])
    art_cfg["init_rot"] = list(mixer.get("init_rot") or [0.0, 0.0, 0.0])
    art_cfg["body_scale"] = list(mixer.get("body_scale") or [1.0, 1.0, 1.0])

    # 3) 事件：reset 时根位姿跟随机摆位后的搅拌器；每控制步切两段驱动
    events = cfg.setdefault("env", {}).setdefault("events", {})
    _insert_after(events, "random_beaker_mixer_pose", "p001_follow_mixer_pose", {
        "func": EVENT_FOLLOW,
        "mode": "reset",
        "params": {"uid": p["uid"], "mixer_uid": p["mixer_uid"]},
    })
    events["p001_switch_toggle"] = {
        "func": EVENT_TOGGLE,
        "mode": "interval",
        "interval_step": 1,
        "params": {
            "uid": p["uid"], "joint": p["joint"], "theta_max_deg": float(p["theta_max_deg"]),
            "hysteresis_deg": float(p["hysteresis_deg"]),
            "contact_sensor": p["contact_sensor"], "lever_link": p["lever_link"],
            "left_links": list(p["left_links"]), "impulse_min": float(p["impulse_min"]),
            "monitor": bool(p["monitor"]),
        },
    }

    # 4) 功能部位（ctx.parts 第 3 版的同名量；框架落地前先走 affordance_datas）
    attrs = {
        "switch_pivot_x": float(PIVOT_SCALED[0]),
        "switch_pivot_y": float(PIVOT_SCALED[1]),
        "switch_pivot_z": float(PIVOT_SCALED[2]),
        "switch_axis": [0.0, 1.0, 0.0],
        "switch_theta_max": math.radians(float(p["theta_max_deg"])),
        "switch_joint": p["joint"],
        "switch_lever_link": p["lever_link"],
    }
    prep = events.get("prepare_extra_attr")
    if prep is None:
        raise RuntimeError("⛔ P001：官方配置里没有 prepare_extra_attr 事件")
    by_name = {a.get("name"): a for a in prep["params"]["attrs"]}
    for k, v in attrs.items():
        if k in by_name:  # 重入（换 θmax）时把值同步过去，别留旧数
            by_name[k]["value"] = v
            by_name[k]["mode"] = "static"
            by_name[k]["entity_cfg"] = {"uid": p["mixer_uid"]}
            continue
        prep["params"]["attrs"].append({
            "name": k, "mode": "static", "entity_cfg": {"uid": p["mixer_uid"]}, "value": v,
        })
    reg = events.get("register_info_to_env")
    if reg is not None:
        for entry in reg["params"]["registry"]:
            if entry.get("entity_cfg", {}).get("uid") == p["mixer_uid"]:
                keys = entry.setdefault("attrs", [])
                for k in attrs:
                    if k not in keys:
                        keys.append(k)

    # 5) 接触传感器：把拨杆加进「指定对象」名单（open question: does the contact sensor accept a new articulation? verify on a simulator run）
    sensors = cfg.get("sensor") or []
    cs = next((s for s in sensors if s.get("uid") == p["contact_sensor"]), None)
    if cs is None:
        raise RuntimeError(f"⛔ P001：官方配置里没有接触传感器 {p['contact_sensor']!r}")
    acfg = cs.setdefault("articulation_cfg_list", [])
    if not any(a.get("articulation_uid") == p["uid"] for a in acfg):
        acfg.append({"articulation_uid": p["uid"], "link_name_list": [p["lever_link"]]})

    return cfg


# ---------------------------------------------------------------- env 侧

def _drive_values(art, jid):
    vals = art.get_joint_drive(joint_ids=[jid])
    vals = list(vals) + [None] * 4
    stiff, damp, effort, vel = vals[:4]
    return {k: float(_np(v).reshape(-1)[0]) for k, v in
            zip(DRIVE_KEYS, (stiff, damp, effort, vel))}


def _contact_state(env, p, art=None, i: int = 0):
    """当步「左手指 × 拨杆」接触。

    返回 dict：{ok(读得到吗), contact, impulse, error, n_left_ids, n_lever_ids}。
    **读不到不是「没接触」**：ok=False 时 error 一定有值，verify() 会拒绝。
    """
    u = _u(env)
    out = {"ok": False, "contact": False, "impulse": 0.0, "error": "",
           "n_left_ids": 0, "n_lever_ids": 0}
    try:
        sensor = u.sim.get_sensor(p["contact_sensor"])
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"no_sensor:{type(ex).__name__}"
        return out
    if sensor is None:
        out["error"] = "no_sensor"
        return out
    try:
        data = sensor.get_data()
        valid = _np(data["is_valid"])
        valid = valid.reshape(valid.shape[0], -1)
        uids = _np(data["user_ids"])
        imp = _np(data["impulse"])
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"bad_sensor_data:{type(ex).__name__}"
        return out
    art = art or _art(env, p)
    try:
        lever_ids = _np(art.get_user_ids(link_name=p["lever_link"])).reshape(-1)
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"no_lever_ids:{type(ex).__name__}"
        return out
    out["n_lever_ids"] = int(lever_ids.size)
    if lever_ids.size == 0:
        out["error"] = "no_lever_ids"
        return out
    robot = getattr(u, "robot", None)
    if robot is None and hasattr(u.sim, "get_robot"):
        robot = u.sim.get_robot("CobotMagic")
    if robot is None:
        out["error"] = "no_robot"
        return out
    left_ids = []
    for ln in p["left_links"]:
        try:
            ids = _np(robot.get_user_ids(link_name=ln)).reshape(-1)
        except Exception:  # noqa: BLE001
            continue
        left_ids.extend(ids.tolist())
    out["n_left_ids"] = len(left_ids)
    if not left_ids:
        out["error"] = "no_left_link_ids"
        return out
    out["ok"] = True
    if not valid.any():
        return out  # 读得到、这步确实没接触
    m = valid[i]
    from .P001_criteria import contact_pairs
    has, best = contact_pairs(uids[i][m], imp[i][m], lever_ids, left_ids)
    out["contact"] = bool(has)
    out["impulse"] = float(best)
    return out


def _set_root_to_mixer(env, p, env_ids=None):
    art = _art(env, p)
    mixer = _mixer(env, p)
    pose = _np(mixer.get_local_pose(to_matrix=True))
    if env_ids is None:
        idx = list(range(len(pose)))
    elif isinstance(env_ids, slice):
        idx = list(range(len(pose)))[env_ids]
    else:
        idx = [int(i) for i in _np(env_ids).reshape(-1)]
    art.set_local_pose(_pose_arg(pose[idx]), env_ids=idx)


def follow_root(env, env_ids, uid="mixer_switch", mixer_uid="beaker_mixer"):
    """reset 事件：把 mixer_switch 根位姿摆到「搅拌器位姿」（体坐标原点相同）。"""
    p = params({"uid": uid, "mixer_uid": mixer_uid})
    _set_root_to_mixer(env, p, env_ids)
    return None


def toggle_step(env, env_ids, uid="mixer_switch", joint="flip", theta_max_deg=30.0,
                hysteresis_deg=2.0, contact_sensor="beaker_mixer_button_contact",
                lever_link="switch_lever", left_links=("left_link7", "left_link8"),
                impulse_min=0.01, monitor=True):
    """interval 事件（每控制步）：按当前 θ 切两段驱动 target，顺手记判据样本。

    两段：θ < θmid − h ⇒ target 0；θ > θmid + h ⇒ target θmax；中间保持上一次。
    片很轻、阻尼小，越过中间点后驱动（0.1 N·m/rad × 0.26 rad ≈ 0.026 N·m）把它拉到开位。
    """
    p = params({"uid": uid, "joint": joint, "theta_max_deg": theta_max_deg,
                "hysteresis_deg": hysteresis_deg, "contact_sensor": contact_sensor,
                "lever_link": lever_link, "left_links": list(left_links),
                "impulse_min": impulse_min})
    u0 = _u(env)  # 状态一律写在底层 env 上：事件回调、apply_env、episode_metrics 看到的是同一个对象
    art = _art(env, p)
    jid = _joint_id(art, joint)
    theta = _np(art.get_qpos())[:, jid]
    theta_max = math.radians(float(theta_max_deg))
    h = math.radians(float(hysteresis_deg))
    mid = 0.5 * theta_max
    state = getattr(u0, "_p001_toggle_state", None)
    if state is None or len(state) != len(theta):
        state = np.zeros(len(theta), dtype=np.int8)  # 0 = 关位目标，1 = 开位目标
        u0._p001_toggle_state = state
    prev = _np(art.get_qpos(target=True))[:, jid]
    new_state = state.copy()
    new_state[theta > mid + h] = 1
    new_state[theta < mid - h] = 0
    target = np.where(new_state == 1, theta_max, 0.0).astype(np.float32)
    # 只在目标真的变了时下发，省得每步都写 GPU
    if np.max(np.abs(prev - target)) > 1e-6 or np.any(new_state != state):
        art.set_qpos(target.reshape(-1, 1), joint_ids=[jid], target=True)
        u0._p001_toggle_state = new_state

    # 动态坏态：逐步锁存「相对开局」的最大位移和速度（中途偏了又归位也逃不掉；首步也量）
    _latch_dynamics(u0, env, p, theta)

    if monitor:
        mon = getattr(u0, "_p001_monitor", None)
        if mon is not None:
            cs = _contact_state(env, p, art)
            for i in range(len(theta)):
                mon.update(theta[i], cs["contact"], cs["impulse"], error=cs["error"])
    return None


def _bad_obs(u0, reason: str):
    """记一次动力学观测无效（逐步锁存，不随后续恢复清零）。"""
    if reason.startswith("theta"):
        u0._p001_theta_invalid = int(getattr(u0, "_p001_theta_invalid", 0)) + 1
    else:
        u0._p001_pose_invalid = int(getattr(u0, "_p001_pose_invalid", 0)) + 1
    u0._p001_dynamics_error = reason


def _latch_dynamics(u0, env, p, theta):
    """锁存机身相对开局的最大位移、机身最大速度、拨杆外端最大速度，以及无效观测。

    每一步都算，取历史最大值（锁存）：中途被推 5 mm 又归位，末态位移是 0，但这里的最大值还是 5 mm。
    首步也有上一帧（apply_env 记的开局位姿/θ），所以首步跳变也算得出来。

    读数无效（位姿 NaN/inf、形状不对、读位姿抛错、θ NaN）也**逐步锁存**：`max(0, NaN)` 会悄悄保留 0，
    所以必须先核有限性；坏观测计数 `pose_invalid` / `theta_invalid` 只增不减，readback 带出去，
    smoke_checks 见过就拒绝——不能靠后面恢复或另一路有效 0 洗掉。
    """
    try:
        pose = _np(_mixer(env, p).get_local_pose(to_matrix=True))
    except Exception as ex:  # noqa: BLE001
        _bad_obs(u0, f"pose_read:{type(ex).__name__}")
        return
    if pose.ndim != 3 or pose.shape[1:] != (4, 4) or not np.isfinite(pose).all():
        _bad_obs(u0, "pose_nonfinite" if pose.shape[-2:] == (4, 4) else "pose_shape")
        return
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    theta_ok = bool(theta_arr.size and np.isfinite(theta_arr).all())
    if not theta_ok:
        _bad_obs(u0, "theta_nonfinite")

    baseline = getattr(u0, "_p001_baseline_mixer", None)
    if baseline is not None and len(baseline) == len(pose):
        shift = float(np.linalg.norm(pose[:, :3, 3] - baseline[:, :3, 3], axis=1).max())
        if not math.isfinite(shift):
            _bad_obs(u0, "shift_nonfinite")
        else:
            u0._p001_body_shift_max = max(float(getattr(u0, "_p001_body_shift_max", 0.0)), shift)
    dt = float(getattr(u0, "_p001_dt", 0.0) or 0.0)
    prev_pose = getattr(u0, "_p001_prev_mixer_pose", None)
    if prev_pose is not None and dt > 0 and len(prev_pose) == len(pose):
        v = float(np.linalg.norm(pose[:, :3, 3] - prev_pose[:, :3, 3], axis=1).max()) / dt
        if not math.isfinite(v):
            _bad_obs(u0, "speed_nonfinite")
        else:
            u0._p001_body_speed_max = max(float(getattr(u0, "_p001_body_speed_max", 0.0)), v)
    y_axis = pose[:, :3, 1]
    # 不带 abs：翻过 90° 要单调（180° 翻转 = 180°，不是 0°）
    tilt = float(np.degrees(np.arccos(np.clip(y_axis[:, 2], -1.0, 1.0))).max())
    u0._p001_tilt_max_deg = max(float(getattr(u0, "_p001_tilt_max_deg", 0.0)), tilt)
    if baseline is not None and len(baseline) == len(pose):
        xy = float(np.linalg.norm(pose[:, :2, 3] - baseline[:, :2, 3], axis=1).max())
        drop = float(baseline[0, 2, 3] - pose[0, 2, 3])
        u0._p001_xy_shift_max_m = max(float(getattr(u0, "_p001_xy_shift_max_m", 0.0)), xy)
        u0._p001_z_drop_max_m = max(float(getattr(u0, "_p001_z_drop_max_m", 0.0)), drop)
    u0._p001_prev_mixer_pose = pose.copy()
    prev_th = getattr(u0, "_p001_prev_theta", None)
    if theta_ok and prev_th is not None and dt > 0 and len(prev_th) == len(theta_arr):
        v_tip = float(np.max(np.abs(theta_arr - prev_th))) * LEVER_ARM_M / dt
        if not math.isfinite(v_tip):
            _bad_obs(u0, "tip_speed_nonfinite")
        else:
            u0._p001_tip_speed_max = max(float(getattr(u0, "_p001_tip_speed_max", 0.0)), v_tip)
    if theta_ok:
        u0._p001_prev_theta = theta_arr.copy()


def apply_env(env, p: dict | None = None) -> dict:
    """reset 之后：摆根位姿、设限位/驱动、清目标、起监视器，返回 readback。"""
    p = params(p)
    u = _u(env)
    art = _art(env, p)
    jid = _joint_id(art, p["joint"])
    theta_max = math.radians(float(p["theta_max_deg"]))

    # 限位：URDF 里是缺省 30°；随机档改了 θmax 就在这里改（并读回）
    lim = _np(art.get_qpos_limits())[:, jid, :]
    if abs(float(lim[0, 1]) - theta_max) > 1e-4 or abs(float(lim[0, 0])) > 1e-6:
        n_env = lim.shape[0]
        art.set_qpos_limits(np.tile(np.array([[[0.0, theta_max]]], dtype=np.float32), (n_env, 1, 1)),
                            joint_ids=[jid])

    # 驱动：配置里写了，这里再核一遍（「开了没生效」是最坏的失败）；一次给全 4 项
    dv = _drive_values(art, jid)
    if any(abs(dv[k] - float(p[k])) > 1e-6 for k in DRIVE_KEYS):
        import torch  # always present on a real run
        n = len(_np(art.get_qpos()))
        art.set_joint_drive(
            stiffness=torch.full((n, 1), float(p["stiffness"])),
            damping=torch.full((n, 1), float(p["damping"])),
            max_effort=torch.full((n, 1), float(p["max_effort"])),
            max_velocity=torch.full((n, 1), float(p["max_velocity"])),
            drive_type="force", joint_ids=[jid],
        )

    # 根位姿跟搅拌器；开局目标 0（关位）。状态一律写在底层 env（u）上
    _set_root_to_mixer(env, p)
    n = len(_np(art.get_qpos()))
    art.set_qpos(np.zeros((n, 1), dtype=np.float32), joint_ids=[jid], target=True)
    u._p001_toggle_state = np.zeros(n, dtype=np.int8)
    u._p001_baseline_mixer = _np(_mixer(env, p).get_local_pose(to_matrix=True)).copy()

    # 动态坏态锁存器：开局位姿/θ 就是「上一帧」，首步跳变也算得出来
    dt = 1.0 / float(getattr(u, "control_freq", 25) or 25)
    u._p001_dt = dt
    u._p001_prev_mixer_pose = u._p001_baseline_mixer.copy()
    u._p001_prev_theta = np.zeros(n, dtype=np.float64)
    u._p001_body_shift_max = 0.0
    u._p001_body_speed_max = 0.0
    u._p001_tip_speed_max = 0.0
    u._p001_tilt_max_deg = 0.0
    u._p001_xy_shift_max_m = 0.0
    u._p001_z_drop_max_m = 0.0
    u._p001_pose_invalid = 0
    u._p001_theta_invalid = 0
    u._p001_dynamics_error = ""

    # 判据监视器（控制步长 = 1 / control_freq）
    if p["monitor"]:
        from .P001_criteria import SwitchMonitor
        u._p001_monitor = SwitchMonitor(dt, theta_max, frac=float(p["on_frac"]),
                                        hold_s=float(p["hold_s"]),
                                        window_s=float(p["hold_s"]),
                                        impulse_min=float(p["impulse_min"]))
    else:
        u._p001_monitor = None

    # 官方判据看得到拨杆（拨杆本来就是搅拌器的一部分）；可关
    if p["official_judge_sees_lever"] and hasattr(u, "_mixer_user_ids"):
        try:
            lever_ids = _np(art.get_user_ids(link_name=p["lever_link"])).reshape(-1).astype(np.int32)
            cur = _np(u._mixer_user_ids).reshape(-1).astype(np.int32)
            merged = np.unique(np.concatenate([cur, lever_ids]))
            try:
                import torch
                u._mixer_user_ids = torch.as_tensor(merged, dtype=torch.int32,
                                                   device=u._mixer_user_ids.device)
            except ImportError:
                u._mixer_user_ids = merged
        except Exception:  # noqa: BLE001
            pass

    return readback(env, p)


def readback(env, p: dict | None = None) -> dict:
    """读回生效值：第一集关卡按 verify() 核，对不上就停链。"""
    p = params(p)
    u = _u(env)
    art = _art(env, p)
    mixer = _mixer(env, p)
    jid = _joint_id(art, p["joint"])
    theta = _np(art.get_qpos())[:, jid]
    lim = _np(art.get_qpos_limits())[:, jid, :]
    tgt = _np(art.get_qpos(target=True))[:, jid]
    root = _np(art.get_local_pose(to_matrix=True))
    mpose = _np(mixer.get_local_pose(to_matrix=True))
    dev_m, dev_deg = _pose_dev(root, mpose)
    base = getattr(u, "_p001_baseline_mixer", None)
    body_shift = None
    xy_shift = None
    z_drop = None
    if base is not None:
        body_shift = float(np.linalg.norm(mpose[:, :3, 3] - base[:, :3, 3], axis=1).max())
        xy_shift = float(np.linalg.norm(mpose[:, :2, 3] - base[:, :2, 3], axis=1).max())
        z_drop = float(base[0, 2, 3] - mpose[0, 2, 3])
    # 搅拌器姿态：本体 **y** 轴（up_axis，字段表；init_rot [90,-90,0]）和世界 z 的夹角
    y_axis = mpose[:, :3, 1]
    # 不带 abs：翻过 90° 要单调（180° 翻转 = 180°，不是 0°）
    tilt = float(np.degrees(np.arccos(np.clip(y_axis[:, 2], -1.0, 1.0))).max())
    cs = _contact_state(env, p, art)
    tip_err, tip_world = _lever_tip_err(env, p, art, theta, mpose)
    out = {
        "patch": PATCH_ID,
        "version": PATCH_VERSION,
        "uid": p["uid"],
        "joint": p["joint"],
        "joint_names": list(art.joint_names),
        "joint_id": jid,
        "theta_rad": [round(float(x), 6) for x in theta],
        "theta_deg": [round(float(np.degrees(x)), 3) for x in theta],
        "theta_max_rad": round(float(lim[0, 1]), 6),
        "theta_min_rad": round(float(lim[0, 0]), 6),
        "theta_max_deg": round(float(np.degrees(lim[0, 1])), 3),
        "target_qpos_rad": [round(float(x), 6) for x in tgt],
        "drive": _drive_values(art, jid),
        "root_pose": root.round(6).tolist(),
        "mixer_pose": mpose.round(6).tolist(),
        "root_dev_m": round(dev_m, 6),
        "root_dev_deg": round(dev_deg, 4),
        "lever_tip_err_m": None if tip_err is None else round(tip_err, 6),
        "lever_tip_world": tip_world,
        "mixer_body_scale": list(mixer.cfg.body_scale) if getattr(mixer, "cfg", None) is not None else None,
        "art_body_scale": list(art.cfg.body_scale) if getattr(art, "cfg", None) is not None else None,
        "body_shift_m": None if body_shift is None else round(body_shift, 6),
        "mixer_tilt_deg": round(tilt, 3),
        "mixer_tilt_max_deg": round(float(getattr(u, "_p001_tilt_max_deg", tilt)), 3),
        "mixer_xy_shift_max_m": round(float(getattr(u, "_p001_xy_shift_max_m", 0.0)), 6),
        "mixer_z_drop_max_m": round(float(getattr(u, "_p001_z_drop_max_m", 0.0)), 6),
        "mixer_xy_shift_m": None if xy_shift is None else round(xy_shift, 6),
        "mixer_z_drop_m": None if z_drop is None else round(z_drop, 6),
        "body_shift_max_m": round(float(getattr(u, "_p001_body_shift_max", 0.0)), 6),
        "body_speed_max_mps": round(float(getattr(u, "_p001_body_speed_max", 0.0)), 6),
        "lever_tip_speed_max_mps": round(float(getattr(u, "_p001_tip_speed_max", 0.0)), 6),
        "pose_invalid_steps": int(getattr(u, "_p001_pose_invalid", 0)),
        "theta_invalid_steps": int(getattr(u, "_p001_theta_invalid", 0)),
        "dynamics_error": str(getattr(u, "_p001_dynamics_error", "")),
        "lever_contact": bool(cs["contact"]),
        "lever_contact_impulse": round(float(cs["impulse"]), 6),
        "contact_ok": bool(cs["ok"]),
        "contact_error": cs["error"],
        "contact_left_ids": cs["n_left_ids"],
        "contact_lever_ids": cs["n_lever_ids"],
        "body_obj": str(BODY_OBJ),
        "urdf": str(URDF),
        "assets_ok": BODY_OBJ.exists() and URDF.exists(),
    }
    return out


def _num(x):
    """把读回值变成 float；不是数就返回 None（NaN/inf 也算无效）。"""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def verify(rb: dict, p: dict | None = None, tol_theta_deg: float = 0.5,
           tol_root_m: float = 0.001, tol_root_deg: float = 0.5,
           tol_tip_m: float = 0.002) -> list:
    """读回对不上就返回问题清单（空 = 过）。第一集关卡用。

    缺字段、形状不对、NaN/inf、接触对象读不到，**一律算不过**——「读不到」不许当「没接触」。
    """
    p = params(p)
    bad = []
    want = math.radians(float(p["theta_max_deg"]))
    if not rb.get("assets_ok"):
        bad.append(f"资产不在：{rb.get('body_obj')} / {rb.get('urdf')}")

    def need(key, label):
        v = _num(rb.get(key))
        if v is None:
            bad.append(f"{label} 读回无效（{key}={rb.get(key)!r}，要有限数）")
        return v

    th_max = need("theta_max_rad", "θmax")
    th_min = need("theta_min_rad", "θmin")
    if th_max is not None and abs(th_max - want) > math.radians(tol_theta_deg):
        bad.append(f"θmax 读回 {th_max:.4f} rad，配置要 {want:.4f} rad")
    if th_min is not None and abs(th_min) > 1e-3:
        bad.append(f"θmin 读回 {th_min:.4f}，应为 0")

    drive = rb.get("drive")
    if not isinstance(drive, dict):
        bad.append(f"drive 读回无效：{drive!r}")
    else:
        for k in DRIVE_KEYS:
            v = _num(drive.get(k))
            if v is None:
                bad.append(f"驱动 {k} 读回无效（{drive.get(k)!r}）")
            elif abs(v - float(p[k])) > 1e-4 * max(1.0, abs(float(p[k]))):
                bad.append(f"驱动 {k} 读回 {v}，配置要 {p[k]}")

    dev_m = need("root_dev_m", "根位姿平移差")
    dev_deg = need("root_dev_deg", "根位姿转角差")
    if dev_m is not None and dev_m > tol_root_m:
        bad.append(f"根位姿离搅拌器 {dev_m:.4f} m，应 < {tol_root_m} m")
    if dev_deg is not None and dev_deg > tol_root_deg:
        bad.append(f"根位姿转角差 {dev_deg:.2f}°，应 < {tol_root_deg}°")

    tip = rb.get("lever_tip_err_m")
    if tip is None:
        bad.append("读不到 switch_lever 的 link 位姿，开关点世界位置没法核")
    else:
        tip_v = _num(tip)
        if tip_v is None:
            bad.append(f"开关点世界位置读回无效（{tip!r}）")
        elif tip_v > tol_tip_m:
            bad.append(f"开关点世界位置差 {tip_v:.4f} m（应 < {tol_tip_m} m）："
                       f"URDF 原始系 + body_scale 没对上，do not trust the run")

    bs, ab = rb.get("mixer_body_scale"), rb.get("art_body_scale")
    for name, v in (("机身", bs), ("关节体", ab)):
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            bad.append(f"{name} body_scale 读回无效（{v!r}，要 3 项）")
    if isinstance(bs, (list, tuple)) and isinstance(ab, (list, tuple)) and len(bs) == 3 and len(ab) == 3:
        for x, y in zip(bs, ab):
            fx, fy = _num(x), _num(y)
            if fx is None or fy is None:
                bad.append(f"body_scale 有非有限值：机身 {bs} / 关节 {ab}")
                break
            if abs(fx - fy) > 1e-6:
                bad.append(f"body_scale 不一致：机身 {bs} / 关节 {ab}")
                break

    if not rb.get("contact_ok"):
        bad.append(f"接触读不到（{rb.get('contact_error') or '未标原因'}）：传感器 {p['contact_sensor']!r}、"
                   f"左手指 id {rb.get('contact_left_ids')} 个、拨杆 id {rb.get('contact_lever_ids')} 个")
    return bad


# 部位表（字段表第 3 版）：存**未缩放网格值**，共用层乘当集 s（0.3 × 尺寸比例）。
# 供 envpatch 框架并进 ctx.parts["mixer_switch"]；专家侧按 flip 的 pivot/axis 把 push_point 转出来。
_PARTS_RAW = {
    "mixer_switch": {
        "flip": {"kind": "joint", "q_name": "flip",
                 "pivot": [-0.35, 0.0, 0.1266667], "axis": [0.0, 1.0, 0.0],
                 "link": "switch_lever", "tol": None, "src": "official"},
        "push_point": {"kind": "axis", "p": [-0.392, 0.0, 0.1866667],
                       "axis": [0.0, 1.0, 0.0], "normal": [0.0, 0.0, 1.0],
                       "size": {"length": 0.0733333}, "lever": 0.0733333,
                       "link": "switch_lever", "tol": None, "src": "official"},
    }
}


def parts(p: dict | None = None) -> dict:
    """返回 mixer_switch 的部位（未缩放网格系）。θmax 按参数写进 flip.range/travel。"""
    p = params(p)
    out = copy.deepcopy(_PARTS_RAW)
    th = math.radians(float(p["theta_max_deg"]))
    flip = out["mixer_switch"]["flip"]
    flip["range"] = [0.0, th]
    flip["travel"] = th
    flip["unit"] = "rad"
    return out


def parts_scaled(parts_dict: dict, s) -> dict:
    """参考实现：把 parts() 的**未缩放值**按字段表第 3 版变换到当集几何（框架可照抄/直接调）。

    - 点 `p' = diag(s)·p`；切向 `axis' = normalize(diag(s)·axis)`；
    - 法向 `normal' = normalize(diag(1/s)·normal)`（逐轴不等缩放时法向要这么变）；
    - 长度量（size.length、lever）沿轴乘 `|diag(s)·axis|`，用**归一化之前**的原始 axis 算。

    前提：`s` 是长度 3 的向量；`axis` 是单位向量（不是单位向量时长度口径没有定义，会报错）。
    输入点/轴/法向均在未缩放网格局部系；返回值仍在物体局部系，不含根位姿。
    size.length/lever 按各自字段的原始 axis 缩放，不从 p 到 pivot 的距离另推力臂。
    """
    import copy as _copy
    import numpy as np

    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if s.size != 3:
        raise ValueError(f"parts_scaled 的 s 必须是长度 3 的向量，收到 {s.size} 个")
    out = _copy.deepcopy(parts_dict)
    for uid, table in out.items():
        for name, part in table.items():
            raw_axis = None
            if "axis" in part:
                raw_axis = np.asarray(part["axis"], dtype=np.float64)
                na = float(np.linalg.norm(raw_axis))
                if abs(na - 1.0) > 1e-6:
                    raise ValueError(f"{uid}.{name}.axis 不是单位向量（|a|={na:.6f}），长度量口径没有定义")
            if "p" in part:
                part["p"] = [float(x) for x in (s * np.asarray(part["p"], dtype=np.float64))]
            if "pivot" in part:
                part["pivot"] = [float(x) for x in (s * np.asarray(part["pivot"], dtype=np.float64))]
            axis_len = None
            if raw_axis is not None:
                a = s * raw_axis                      # |diag(s)·axis| 用这一份（归一化之前）
                axis_len = float(np.linalg.norm(a))
                part["axis"] = [float(x) for x in (a / axis_len)]
            if "normal" in part:
                nn = np.asarray(part["normal"], dtype=np.float64) / np.where(s == 0, 1.0, s)
                ln = float(np.linalg.norm(nn))
                part["normal"] = [float(x) for x in (nn / ln if ln > 0 else nn)]
            if axis_len is not None:
                if "size" in part and "length" in part["size"]:
                    part["size"]["length"] = float(part["size"]["length"] * axis_len)
                if "lever" in part:
                    part["lever"] = float(part["lever"] * axis_len)
    return out


def push_point_at(parts_dict: dict, s, theta: float) -> dict:
    """Reference for consumers: 当集几何（乘 s）下、关节角 θ 时的开关点 p 和片外法向 n。

    用 flip 的 pivot/axis 把 θ=0 的 push_point 转出来（右手绕 axis 转 θ）。
    parts_dict 必须是 parts() 的未缩放局部表，本函数先缩放一次，再在缩放后的局部系转动；
    不接收 parts_scaled() 的输出，避免重复缩放。theta 用 rad，由调用方检查关节范围。
    返回 p/normal 仍是物体局部系：转世界时点用 R·p+t，法向只用 R·normal。
    非均匀缩放与转动不交换；这里约定先缩放，再作刚体关节转动。
    """
    import numpy as np

    sc = parts_scaled(parts_dict, s)["mixer_switch"]
    flip, push = sc["flip"], sc["push_point"]
    piv = np.asarray(flip["pivot"], dtype=np.float64)
    ax = np.asarray(flip["axis"], dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    rel = np.asarray(push["p"], dtype=np.float64) - piv
    # Rodrigues
    c, s_ = math.cos(theta), math.sin(theta)
    rot = (rel * c + np.cross(ax, rel) * s_ + ax * np.dot(ax, rel) * (1 - c))
    n0 = np.asarray(push["normal"], dtype=np.float64)
    n = (n0 * c + np.cross(ax, n0) * s_ + ax * np.dot(ax, n0) * (1 - c))
    return {"p": [float(x) for x in (piv + rot)], "normal": [float(x) for x in (n / np.linalg.norm(n))],
            "lever": float(push["lever"]), "theta": float(theta)}


def episode_metrics(env) -> dict | None:
    mon = getattr(_u(env), "_p001_monitor", None)
    return None if mon is None else mon.metrics()


def smoke_checks(*, n_steps: int, monitor, rb_final: dict, max_body_speed_mps=None,
                 max_lever_tip_speed_mps=None, p: dict | None = None,
                 require_flip: bool = True, min_theta_deg: float = 0.5,
                 body_shift_max_m: float = 0.003, speed_max_mps: float = 1.0,
                 final_readback: bool = False, max_body_shift_m=None,
                 pose_invalid_steps=None, theta_invalid_steps=None) -> list:
    """冒烟通过条件（纯函数，离线可测）。返回问题清单，空 = 过。

    Failures must be loud: 悄悄失效、异常速度、机身被推走，都算不过，不许只打一行日志就放行。

    位移/速度取**全程锁存的最大值**（rb 里的 `body_shift_max_m` / `body_speed_max_mps` /
    `lever_tip_speed_max_mps`，由补丁每步更新；调用方也可以另外传一份逐步算的，取两者最大）。
    只看末态会把「中途被推 5 mm 又归位」漏掉；只看第二步以后会把首步跳变漏掉。
    final_readback=True 时跳过「根位姿离搅拌器」这一项：回合末机身被推走多少由位移最大值判。
    """
    p = params(p)
    bad = []
    if monitor is None:
        return ["监视器没挂上：interval 事件没收到样本（观测缺失）"]
    if n_steps <= 0:
        bad.append("一集 0 步，没东西可判")
    if len(monitor.theta) != n_steps:
        bad.append(f"逐步样本 {len(monitor.theta)} 条 ≠ 跑了 {n_steps} 步（观测缺失）")
    if monitor.contact_error_steps:
        bad.append(f"接触有 {monitor.contact_error_steps} 步读不到（{monitor.last_contact_error}）")
    th = [float(x) for x in monitor.theta]
    if not th or any(not math.isfinite(x) for x in th):
        bad.append("θ 序列有非有限值")
    elif max(abs(x) for x in th) < math.radians(min_theta_deg):
        bad.append(f"θ 全程 < {min_theta_deg}°：两段驱动没接上，或手没碰到拨杆")
    if not any(monitor.contact):
        bad.append("全程没读到「左手指 × 拨杆」接触：传感器/link 名没生效，或手没碰拨杆")
    m = monitor.metrics()
    if require_flip:
        if not m.get("switched_on"):
            bad.append(f"没停住：flip_frac_max={m.get('flip_frac_max')} flip_hold_s={m.get('flip_hold_s')}")
        if not m.get("flip_by_left"):
            bad.append(f"不是左手推的：cross_step={m.get('flip_cross_step')} "
                       f"impulse={m.get('flip_contact_impulse')}")

    def _max_of(*vals):
        got = [float(v) for v in vals if v is not None and _num(v) is not None]
        return max(got) if got else None

    pi = _max_of(rb_final.get("pose_invalid_steps"), pose_invalid_steps) or 0.0
    ti = _max_of(rb_final.get("theta_invalid_steps"), theta_invalid_steps) or 0.0
    if pi > 0:
        bad.append(f"中途有 {int(pi)} 步机身位姿无效（{rb_final.get('dynamics_error') or '未标原因'}）"
                   f"——坏观测锁存，不许被后面的正常值洗掉")
    if ti > 0:
        bad.append(f"中途有 {int(ti)} 步 θ 无效（{rb_final.get('dynamics_error') or '未标原因'}）")

    # 机身位移：**只报不判**（3 mm 是我们自己加的，FAQ / 官方判据都没有）
    shift = _max_of(rb_final.get("body_shift_max_m"), rb_final.get("body_shift_m"), max_body_shift_m)
    if shift is None:
        bad.append("机身挪动读不到（body_shift_max_m / body_shift_m / 逐步值都没有）")
    # 硬项只留「搅拌器没被推倒 / 推出台面」
    tilt = _max_of(rb_final.get("mixer_tilt_max_deg"), rb_final.get("mixer_tilt_deg"))
    xy = _max_of(rb_final.get("mixer_xy_shift_max_m"), rb_final.get("mixer_xy_shift_m"))
    drop = _max_of(rb_final.get("mixer_z_drop_max_m"), rb_final.get("mixer_z_drop_m"))
    if tilt is None or xy is None or drop is None:
        bad.append(f"搅拌器姿态读不到（tilt={tilt} xy={xy} z_drop={drop}）")
    else:
        if tilt > 45.0:
            bad.append(f"搅拌器被推倒：倾角 {tilt:.1f}° > 45°")
        if drop > 0.02:
            bad.append(f"搅拌器被推下台面：原点比开局低 {drop * 1000:.0f} mm")
        if xy > 0.5:
            bad.append(f"搅拌器被推出台面：水平挪 {xy:.2f} m > 0.5 m")
    for name, latched, passed in (("机身速度", rb_final.get("body_speed_max_mps"), max_body_speed_mps),
                                  ("拨杆外端速度", rb_final.get("lever_tip_speed_max_mps"),
                                   max_lever_tip_speed_mps)):
        v = _max_of(latched, passed)
        if v is None:
            bad.append(f"{name}读不到（{latched!r} / {passed!r}）")
        elif v > speed_max_mps:
            bad.append(f"{name} {v:.2f} m/s > {speed_max_mps} m/s（标「飞出」，关补丁）")
    rb = dict(rb_final)
    if final_readback:
        rb["root_dev_m"] = 0.0
        rb["root_dev_deg"] = 0.0
    bad.extend(verify(rb, p))
    return bad
