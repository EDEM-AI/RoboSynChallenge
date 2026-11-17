"""Gate 7 "environment anomaly": offline implementation of rules R0-R8，读 rsrec 记录（schema v1 = rsrec/1.2；v1.1 也能读，缺的字段记 unknown）。

阈值等级 A = 既有代码 / 官方判据 / 旧实证，B = 工程建议、没标定。
**B 级只告警（黄），不自动丢集**；`--strict` 把黄当红（停批判断用）。确认的物理异常才隔离，评测主分母不减。

  R0 记录有效性     非有限 / 缺值位姿、四元数范数误差 > 1e-3（含全 0 位姿，E19）、t / sim_t 不递增               → 红（数据坏，不算环境自运动）
  R1 无作用刚体移动  原来静止（前 0.2 s 线速 < 0.01 m/s、角速 < 0.1 rad/s）+ 机器人远（双法兰 > 0.25 m，只能当代理）
                    + 附近没有别的物体在动 ⇒ 0.2 s 窗内位移 > 5 mm 或转角 > 3°、持续 ≥ 0.08 s；单步 > 2 cm / 10° 另报 jump
                    开局 0.5 s 内的另记 settling                                                                   → 黄（没有接触列 ⇒ 不能确认）
  R2 关节自开       开局 q0 − q_closed > 5 mm（A，#80）→ 红；中途机器人远 + 0.2 s 内 qmax − qmin > 5 mm → 黄           （主动按的按钮除外）
  R3 速度突跳       差分速度减重力后 |Δv| > max(0.5 m/s, 30 m/s²·dt) 或一步位移 > 0.05 m；cause = 机器人远 ⇒ uncaused、近 ⇒ unknown → 黄
  R4 穿桌 / 掉桌    缩放后包围盒最低点低于桌面 > 5 mm 且 xy 在桌内、持续 ≥ 0.08 s → penetration；整体低于桌面 2 cm → below_table；
                    xy 出桌且下降 > 5 cm → fall_off（多半是策略 / 接触失败）                                           → 黄；没 geom ⇒ unknown
  R5 动态不自洽     要逐步速度（rich 子集才有）：|Δp − (v_t + v_{t-1})·dt/2| > max(5 mm, 0.2·|Δp|)；关节越限                → 黄；没速度 ⇒ unknown
  R6 判据不一致     开局（第 0 行）或机器人还没合过爪就成功 ⇒ success_without_action；官方 / 原样复算 / 独立语义三套标签的比较要任务插件
                    （VERIFIERS，这版没有 ⇒ unknown + 缺字段）                                                       → 黄
  R7 同种子漂移     同种子 + 同版本 + 同配置的开局：物体位差 > 1 mm 或转角 > 0.5°、机器人关节 > 1e-3 rad ⇒ pair_mismatch     → 黄（不丢训练，只是不能当严格配对）
  R8 光照 / 外观事件 --appearance reset-only 时集内任何外观 setter ⇒ 红；interval（官方默认）时报实际触发步，步距全是 10 的倍数
                    或首次触发晚于第 0 步（前段默认灯）⇒ 黄；不和 R1–R5 合成物理原因

每条事件：{rule_id, sub, episode, seed, uid, t_start, t_end, observed, threshold, unit, evidence_level, contact_coverage, cause,
           missing_fields, official_s, verified_s, level}；每集分 opening_flags / mid_episode_flags / criterion_flags /
           data_integrity_flags / appearance_flags 五组，另列 unknown（缺字段算不了的）。
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
import sys
import zlib
from collections import defaultdict

import numpy as np

RED, YELLOW, GREEN, INFO = "红", "黄", "绿", "注"   # 注 = 记下、不影响颜色（机器人近时的突跳 / 掉桌：归接触或策略，不算环境异常）
G = 9.81
TH = dict(
    quat_tol=1e-3, static_win_s=0.2, static_v=0.01, static_w=0.1, far_m=0.25, near_other_m=0.10,
    r1_disp_m=0.005, r1_rot_deg=3.0, r1_hold_s=0.08, r1_jump_m=0.02, r1_jump_deg=10.0, settle_s=0.5, pre_s=0.5,
    r2_q0_m=0.005, r2_mid_m=0.005, r2_far_m=0.30,
    r3_dv=0.5, r3_acc=30.0, r3_step_m=0.05, r3_dw=5.0, r3_alpha=300.0,
    r4_pen_m=0.005, r4_below_m=0.02, r4_fall_m=0.05, r4_margin=0.01, r4_hold_s=0.08,
    r5_m=0.005, r5_rel=0.2, r5_lim_m=1e-4,
    r7_pos=0.001, r7_rot_deg=0.5, r7_q=1e-3,
    hidden_z=-1.0, stop_frac=0.05,
)
# 主动由机器人驱动的关节（按铃盖、移液器按钮）不套 R2 自开
ACTIVE_ARTS = {"click_bell": {"button"}, "manipulate_pipette": {"pipette"}}
APPEAR_CLS = {"Light", "VisualMaterialInst", "SimulationManager"}
APPEAR_METH = {"set_visual_material"}
VERIFIERS = {}   # 任务 → fn(ep, S) → (replay_s, verified_s)；这版没有（R6 三套标签比较的接口）
GROUPS = ("opening_flags", "mid_episode_flags", "criterion_flags", "data_integrity_flags", "appearance_flags")


# ------------------------------------------------------------------ 读
def read_lines(path):
    out = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except (EOFError, zlib.error, OSError):
        pass
    return out


def load_episode(path):
    ep = dict(path=str(path), header=None, steps=[], footer=None, plans=[])
    for ln in read_lines(path):
        k = ln.get("kind")
        if k == "header":
            ep["header"] = ln
        elif k == "step":
            ep["steps"].append(ln)
        elif k == "footer":
            ep["footer"] = ln
        elif k == "plan":
            ep["plans"].append(ln)
    uids = ((ep.get("header") or {}).get("ffill") or {}).get("uids") or []     # v1：隐藏池没变的列不写 ⇒ 按上一行补回
    if uids:
        pre, last = tuple(f"obj_{u}_" for u in uids), {}
        for s in ep["steps"]:
            st = s.get("state")
            if st is None:
                continue
            last.update({k: v for k, v in st.items() if k.startswith(pre)})
            for k, v in last.items():
                st.setdefault(k, v)
    return ep


def load_run(run_dir):
    fs = sorted(glob.glob(os.path.join(run_dir, "ep_*.jsonl.gz"))) + sorted(glob.glob(os.path.join(run_dir, "ep_*.jsonl.gz.part")))
    return [load_episode(p) for p in fs]


# ------------------------------------------------------------------ 取数
AXES = ("x", "y", "z", "qw", "qx", "qy", "qz")


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def series(ep):
    rows = [s for s in ep["steps"] if s.get("state") is not None]
    st = [s["state"] for s in rows]
    keys = set().union(*[set(x) for x in st]) if st else set()
    ent = {}
    for k in keys:
        for pre in ("obj_", "art_"):
            if k.startswith(pre) and k.endswith("_qw") and "_qpos_" not in k:
                uid = k[len(pre):-3]
                ent[uid] = dict(kind=pre[:-1], a=np.array([[_num(x.get(f"{pre}{uid}_{a}")) for a in AXES] for x in st], float))
    artq = defaultdict(dict)
    for k in keys:
        if k.startswith("art_") and "_qpos_" in k:
            uid, j = k[4:].rsplit("_qpos_", 1)
            artq[uid][int(j)] = np.array([_num(x.get(k)) for x in st], float)
    fl = {h: np.array([[_num(x.get(f"link_{h}_eef_{a}")) for a in "xyz"] for x in st], float) for h in ("left", "right")
          if f"link_{h}_eef_x" in keys}
    grip = {h: np.array([_num(x.get(f"qpos_{h}_eef_0")) for x in st], float) for h in ("left", "right") if f"qpos_{h}_eef_0" in keys}
    t = np.array([s.get("t", np.nan) for s in rows], float)
    h = ep.get("header") or {}
    dt0 = float(((h.get("timing") or {}).get("dt_s")) or 1.0 / float(h.get("rate_hz") or 25.0))
    sim_t = np.array([_num(s.get("sim_t")) if s.get("sim_t") is not None else np.nan for s in rows], float)
    if np.isnan(sim_t).all():
        sim_t = t * dt0
    succ = np.array([_num(x.get("succ")) for x in st], float)
    vel = {}
    if any("vel" in s for s in rows):
        for s_i, s in enumerate(rows):
            for uid, v in (s.get("vel") or {}).items():
                vel.setdefault(uid, np.full((len(rows), 6), np.nan))[s_i] = [_num(x) for x in v[:6]]
    contacts = [s.get("contacts") for s in rows] if any("contacts" in s for s in rows) else None
    return dict(rows=rows, t=t, sim_t=sim_t, dt0=dt0, ent=ent, artq=dict(artq), flange=fl, grip=grip, succ=succ, vel=vel or None,
                contacts=contacts, n=len(rows))


def quat_to_R(q):
    w, x, y, z = q / (np.linalg.norm(q) or 1.0)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_deg(q1, q2):
    n1, n2 = np.linalg.norm(q1), np.linalg.norm(q2)
    if not n1 or not n2 or np.isnan(n1) or np.isnan(n2):
        return np.nan
    return float(np.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q1 / n1, q2 / n2)))))))


def _ev(rule, sub, ep, uid, t0, t1, obs, thr, unit, lvl, level, cov, cause=None, missing=None, **kw):
    h = ep.get("header") or {}
    d = dict(rule_id=rule, sub=sub, episode=h.get("episode"), seed=h.get("seed"), uid=uid, t_start=None if t0 is None else int(t0),
             t_end=None if t1 is None else int(t1), observed=obs if not isinstance(obs, float) else round(obs, 6), threshold=thr, unit=unit,
             evidence_level=lvl, contact_coverage=cov, cause=cause, missing_fields=missing or [], official_s=None, verified_s=None, level=level)
    d.update(kw)
    return d


def _coverage(S):
    return "有接触列" if S["contacts"] is not None else "无接触列：只有双法兰中心（只能黄色候选）"


def _visible(S, uid):
    a = S["ent"][uid]["a"]
    z = a[:, 2]
    return not (np.nanmax(z) < TH["hidden_z"]) if np.isfinite(z).any() else False


def _eef_dist(S, p):
    """每步双法兰到点 p (T,3) 的最近距离；没有法兰 ⇒ inf。"""
    if not S["flange"]:
        return np.full(len(p), np.inf)
    d = [np.linalg.norm(F - p, axis=1) for F in S["flange"].values()]
    return np.nanmin(np.vstack(d), axis=0)


def _dts(S):
    d = np.diff(S["sim_t"])
    d = np.where(np.isfinite(d) & (d > 0), d, S["dt0"])
    return d


def _speeds(S, a):
    """差分线速 / 角速（第 i 个 = 第 i-1 → i 步）。"""
    dts = _dts(S)
    v = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1) / dts
    w = np.array([rot_deg(a[i, 3:7], a[i + 1, 3:7]) for i in range(len(a) - 1)]) * math.pi / 180 / dts
    return np.r_[np.nan, v], np.r_[np.nan, w]


def _reset_rows(S):
    return {i for i, s in enumerate(S["rows"]) if s.get("auto_reset")}


# ------------------------------------------------------------------ R0
def r0(ep, S):
    out = []
    for uid, e in S["ent"].items():
        a = e["a"]
        bad = np.where(np.isnan(a).any(axis=1))[0]
        if len(bad):
            out.append(_ev("R0", "nonfinite", ep, uid, S["t"][bad[0]], S["t"][bad[-1]], int(len(bad)), "任一非有限 / 缺值", "步", "B", RED, _coverage(S)))
        qn = np.linalg.norm(a[:, 3:7], axis=1)
        badq = np.where(np.isfinite(qn) & (np.abs(qn - 1.0) > TH["quat_tol"]))[0]
        if len(badq):
            out.append(_ev("R0", "quat_norm", ep, uid, S["t"][badq[0]], S["t"][badq[-1]], float(np.nanmax(np.abs(qn[badq] - 1))),
                           TH["quat_tol"], "|‖q‖−1|", "B", RED, _coverage(S), cause="全 0 位姿（E19 关节体当刚体读）" if (qn[badq] < 1e-6).any() else None))
    for name, arr in (("t", S["t"]), ("sim_t", S["sim_t"])):
        d = np.diff(arr)
        bad = np.where(np.isfinite(d) & (d <= 0))[0]
        if len(bad):
            out.append(_ev("R0", f"{name}_not_increasing", ep, None, S["t"][bad[0]], S["t"][bad[0] + 1], int(len(bad)), "严格递增", "步", "B", RED, _coverage(S)))
    return out


# ------------------------------------------------------------------ R1 / R3
def _others_moving(S, uid, lo, hi, v_all):
    """[lo, hi] 步内，离 uid 不到 near_other_m（+ 物体尺寸余量 0.1 m）的别的物体有没有在动 ⇒ 外来碰撞，不算自运动。"""
    for i in range(lo, hi + 1):
        p = S["ent"][uid]["a"][i, :3]
        for u, e in S["ent"].items():
            if u == uid or not _visible(S, u):
                continue
            q = e["a"][i, :3]
            if np.isfinite(q).all() and np.isfinite(p).all() and np.linalg.norm(q - p) < TH["near_other_m"] + 0.1 and v_all[u][i] > TH["static_v"]:
                return u
    return None


def r1_r3(ep, S):
    out = []
    n = S["n"]
    if n < 3:
        return out
    cov = _coverage(S)
    dts = _dts(S)
    dt = float(np.median(dts))
    w_static = max(1, int(round(TH["static_win_s"] / dt)))
    w_pre = max(1, int(round(TH["pre_s"] / dt)))
    w_hold = max(1, int(math.ceil(TH["r1_hold_s"] / dt - 1e-9)))
    settle_n = int(round(TH["settle_s"] / dt))
    resets = _reset_rows(S)
    sp = {u: _speeds(S, e["a"]) for u, e in S["ent"].items()}
    v_all = {u: sp[u][0] for u in sp}
    for uid, e in S["ent"].items():
        if e["kind"] != "obj" or not _visible(S, uid):
            continue
        a = e["a"]
        v, w = sp[uid]
        far = _eef_dist(S, a[:, :3]) > TH["far_m"]
        # ---- R1
        i = w_static
        while i < n - 1:
            pre = slice(i - w_static + 1, i + 1)
            static = np.nanmax(v[pre]) < TH["static_v"] and np.nanmax(w[pre]) < TH["static_w"] if i - w_static + 1 >= 1 else False
            if not static:
                i += 1
                continue
            j1 = min(n - 1, i + w_static)
            disp = np.linalg.norm(a[i + 1:j1 + 1, :3] - a[i, :3], axis=1)
            rot = np.array([rot_deg(a[i, 3:7], a[k, 3:7]) for k in range(i + 1, j1 + 1)])
            over = (disp > TH["r1_disp_m"]) | (rot > TH["r1_rot_deg"])
            run, best = 0, 0
            for o in over:
                run = run + 1 if o else 0
                best = max(best, run)
            if best >= w_hold:
                k0 = i + 1 + int(np.argmax(over))
                lo = max(0, i - w_pre)
                gate_far = bool(far[lo:j1 + 1].all())
                other = _others_moving(S, uid, lo, j1, v_all)
                if gate_far and other is None and not (set(range(i, j1 + 1)) & resets):
                    sub = "settling" if i < settle_n else "self_motion_candidate"
                    out.append(_ev("R1", sub, ep, uid, S["t"][k0], S["t"][j1], float(max(np.nanmax(disp), 0.0)),
                                   dict(disp_m=TH["r1_disp_m"], rot_deg=TH["r1_rot_deg"], hold_s=TH["r1_hold_s"]), "m", "A（5 mm）/ B（窗、角、持续）",
                                   YELLOW, cov, cause="机器人远（代理）、附近无别的物体在动", rot_deg=round(float(np.nanmax(rot)), 3),
                                   group="opening_flags" if sub == "settling" else "mid_episode_flags"))
                i = j1 + 1
                continue
            i += 1
        # 开局 settling：头 0.5 s 内离第 0 行 > 5 mm / 3°（不要求原来静止；不豁免开局穿透，R4 另查）
        if settle_n > 1 and far[:settle_n].all():
            d0 = np.linalg.norm(a[1:settle_n, :3] - a[0, :3], axis=1)
            r0_ = np.array([rot_deg(a[0, 3:7], a[k, 3:7]) for k in range(1, settle_n)])
            if (np.nanmax(d0) > TH["r1_disp_m"]) or (np.nanmax(r0_) > TH["r1_rot_deg"]):
                k0 = 1 + int(np.argmax((d0 > TH["r1_disp_m"]) | (r0_ > TH["r1_rot_deg"])))
                out.append(_ev("R1", "settling", ep, uid, S["t"][k0], S["t"][settle_n - 1], float(np.nanmax(d0)),
                               dict(disp_m=TH["r1_disp_m"], rot_deg=TH["r1_rot_deg"], window_s=TH["settle_s"]), "m", "A（5 mm）/ B", YELLOW, cov,
                               cause="开局 0.5 s 内还在动（没落定就开录）", rot_deg=round(float(np.nanmax(r0_)), 3), group="opening_flags"))
        # 单步 jump（R1 附报）
        for k in range(1, n):
            if k in resets or k - 1 in resets:
                continue
            dp = float(np.linalg.norm(a[k, :3] - a[k - 1, :3]))
            dr = rot_deg(a[k - 1, 3:7], a[k, 3:7])
            if (dp > TH["r1_jump_m"] or (dr == dr and dr > TH["r1_jump_deg"])) and far[k - 1] and far[k] and np.nanmax(v[max(1, k - w_static):k]) < TH["static_v"]:
                out.append(_ev("R1", "jump", ep, uid, S["t"][k], S["t"][k], dp, dict(step_m=TH["r1_jump_m"], step_deg=TH["r1_jump_deg"]), "m", "B",
                               YELLOW, cov, cause="静止中单步大跳、机器人远", rot_deg=round(float(dr), 3), group="mid_episode_flags"))
        # ---- R3 速度突跳
        vel = np.diff(a[:, :3], axis=0) / dts[:, None]
        for k in range(1, len(vel)):
            r = k + 1
            if r in resets or k in resets or k == 0:
                continue
            dv = vel[k] - vel[k - 1] - np.array([0, 0, -G]) * dts[k]
            ndv = float(np.linalg.norm(dv))
            # 一步位移：减去「上一步速度 + 重力」预测的位移（自由落体不算跳）
            pred = vel[k - 1] * dts[k] + 0.5 * np.array([0, 0, -G]) * dts[k] ** 2
            step = float(np.linalg.norm(a[r, :3] - a[r - 1, :3] - pred))
            thr = max(TH["r3_dv"], TH["r3_acc"] * dts[k])
            if (np.isfinite(ndv) and ndv > thr and np.linalg.norm(vel[k - 1]) < 1e3) or step > TH["r3_step_m"]:
                d_e = _eef_dist(S, a[:, :3])[r]
                cause = "uncaused" if d_e > TH["far_m"] else "unknown"
                if S["contacts"] is not None:
                    c = S["contacts"][r] or []
                    cause = "contact_related" if any(uid in pr for pr in c) else "uncaused"
                out.append(_ev("R3", "velocity_jump" if step <= TH["r3_step_m"] else "step_jump", ep, uid, S["t"][r], S["t"][r],
                               round(max(ndv, step), 6), dict(dv=f"max({TH['r3_dv']}, {TH['r3_acc']}·dt)", step_m=TH["r3_step_m"]),
                               "m/s 或 m", "B（0.48 m/步旧实证 A）", YELLOW if cause == "uncaused" else INFO, cov, cause=cause, eef_dist_m=round(float(d_e), 4), group="mid_episode_flags"))
    return _dedup(out)


def _dedup(evs):
    seen, out = set(), []
    for e in evs:
        k = (e["rule_id"], e["sub"], e["uid"], e["t_start"])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


# ------------------------------------------------------------------ R2
def r2(ep, S):
    out = []
    h = ep.get("header") or {}
    task = h.get("task")
    lim = h.get("limits") or {}
    cov = _coverage(S)
    dt = float(np.median(_dts(S))) if S["n"] > 1 else S["dt0"]
    w = max(1, int(round(TH["static_win_s"] / dt)))
    for uid, js in S["artq"].items():
        if uid in ACTIVE_ARTS.get(task, set()):
            continue
        root = S["ent"].get(uid, {}).get("a")
        far = _eef_dist(S, root[:, :3]) > TH["r2_far_m"] if root is not None else np.zeros(S["n"], bool)
        for j, q in js.items():
            lo = (lim.get(uid) or {}).get("lo")
            q_closed = float(lo[j]) if lo and j < len(lo) else 0.0
            src = "header.limits" if lo and j < len(lo) else "假定 0（没读到限位）"
            off = float(q[0] - q_closed) if np.isfinite(q[0]) else np.nan
            if off == off and off > TH["r2_q0_m"]:
                out.append(_ev("R2", "opening_offset", ep, uid, S["t"][0], S["t"][0], off, TH["r2_q0_m"], "m（平移关节）", "A（#80、opening.py）", RED, cov,
                               cause=f"开局已开；q_closed 来自 {src}", joint=j, group="opening_flags"))
            for i in range(w, S["n"]):
                win = slice(i - w, i + 1)
                if not far[win].all():
                    continue
                rng = float(np.nanmax(q[win]) - np.nanmin(q[win]))
                closed_then_open = bool(np.nanmin(q[:i - w + 1] - q_closed) < TH["r2_q0_m"] and (q[win] - q_closed > 2 * TH["r2_mid_m"]).all())
                if rng > TH["r2_mid_m"] or closed_then_open:
                    out.append(_ev("R2", "self_open", ep, uid, S["t"][i - w], S["t"][i], rng if rng > TH["r2_mid_m"] else float(np.nanmin(q[win]) - q_closed),
                                   dict(range_m=TH["r2_mid_m"], hold_open_m=2 * TH["r2_mid_m"], window_s=TH["static_win_s"]), "m / 0.2 s", "B", YELLOW, cov,
                                   cause="机器人远（代理）时关节在动" if rng > TH["r2_mid_m"] else "闭合过、之后 0.2 s 一直开着 > 1 cm、机器人远",
                                   joint=j, q_closed=q_closed, group="mid_episode_flags"))
                    break
    return out


# ------------------------------------------------------------------ R4
def _corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def r4(ep, S):
    out, unk = [], []
    h = ep.get("header") or {}
    g = h.get("geom") or {}
    tab = g.get("table")
    cov = _coverage(S)
    if not tab:
        return out, [dict(rule_id="R4", missing_fields=["header.geom.table"], note="没有桌面几何（v1.1 文件或读不到桌面顶点）")]
    top = float(tab["top_z"])
    xy0, xy1 = np.array(tab["xy_min"]) + TH["r4_margin"], np.array(tab["xy_max"]) - TH["r4_margin"]
    dt = float(np.median(_dts(S))) if S["n"] > 1 else S["dt0"]
    hold = max(1, int(math.ceil(TH["r4_hold_s"] / dt - 1e-9)))
    for uid, e in S["ent"].items():
        if e["kind"] != "obj" or not _visible(S, uid):
            continue
        gg = (g.get("rigid") or {}).get(uid) or {}
        bb = gg.get("aabb_local")
        if not bb:
            unk.append(dict(rule_id="R4", uid=uid, missing_fields=[f"header.geom.rigid.{uid}"]))
            continue
        # 优先用 26 方向支撑点（网格上的真顶点）；老文件只有包围盒 ⇒ 用角点（斜放时会偏低，结果标 aabb）
        C = np.asarray(gg["support"], float) if gg.get("support") else _corners(*bb)
        a = e["a"]
        bot, topz = np.full(S["n"], np.nan), np.full(S["n"], np.nan)
        for i in range(S["n"]):
            if np.isnan(a[i]).any():
                continue
            W = C @ quat_to_R(a[i, 3:7]).T + a[i, :3]
            bot[i], topz[i] = W[:, 2].min(), W[:, 2].max()
        inxy = (a[:, 0] >= xy0[0]) & (a[:, 0] <= xy1[0]) & (a[:, 1] >= xy0[1]) & (a[:, 1] <= xy1[1])
        pen = inxy & (bot < top - TH["r4_pen_m"])
        run = 0
        for i in range(S["n"]):
            run = run + 1 if pen[i] else 0
            if run == hold:
                k0 = i - hold + 1
                below = bool((topz[k0:i + 1] < top - TH["r4_below_m"]).all())
                out.append(_ev("R4", "below_table" if below else "penetration", ep, uid, S["t"][k0], S["t"][i], float(top - np.nanmin(bot[k0:i + 1])),
                               TH["r4_below_m"] if below else TH["r4_pen_m"], "m（低于桌面）", "B", YELLOW, cov,
                               cause="xy 在桌内、最低点低于桌面（容器内放置在桌面以上，不受影响）", table_top_z=top,
                               geom_src="support" if gg.get("support") else "aabb 角点（斜放会偏低）",
                               group="opening_flags" if k0 == 0 else "mid_episode_flags"))
                break
        out_xy = ~inxy
        z0 = a[0, 2]
        for i in range(S["n"]):
            if out_xy[i] and np.isfinite(a[i, 2]) and z0 - a[i, 2] > TH["r4_fall_m"]:
                d_e = float(_eef_dist(S, a[:, :3])[max(0, i - 5):i + 1].min())
                out.append(_ev("R4", "fall_off", ep, uid, S["t"][i], S["t"][i], float(z0 - a[i, 2]), TH["r4_fall_m"], "m（下降）", "B",
                               INFO if d_e < TH["far_m"] else YELLOW, cov,
                               cause="机器人近（多半是推碰 / 接触失败）" if d_e < TH["far_m"] else "机器人远", group="mid_episode_flags"))
                break
    return out, unk


# ------------------------------------------------------------------ R5
def r5(ep, S):
    out, unk = [], []
    cov = _coverage(S)
    if S["vel"] is None:
        unk.append(dict(rule_id="R5", missing_fields=["step.vel（rich 子集才记）"], note="没有逐步速度，动态自洽算不了"))
    else:
        dts = _dts(S)
        for uid, V in S["vel"].items():
            if uid not in S["ent"]:
                continue
            a = S["ent"][uid]["a"]
            for i in range(1, S["n"]):
                dp = a[i, :3] - a[i - 1, :3]
                pred = (V[i, :3] + V[i - 1, :3]) * dts[i - 1] / 2
                err = float(np.linalg.norm(dp - pred))
                if np.isfinite(err) and err > max(TH["r5_m"], TH["r5_rel"] * float(np.linalg.norm(dp))):
                    out.append(_ev("R5", "state_inconsistent", ep, uid, S["t"][i], S["t"][i], err, f"max({TH['r5_m']}, {TH['r5_rel']}·|Δp|)", "m", "B",
                                   YELLOW, cov, cause="位姿差分和速度梯形积分对不上（先查采样是否同一时点）", group="data_integrity_flags"))
                    break
    lim = (ep.get("header") or {}).get("limits") or {}
    for uid, js in S["artq"].items():
        L = lim.get(uid)
        if not L:
            continue
        for j, q in js.items():
            if j < len(L["lo"]):
                lo, hi = L["lo"][j], L["hi"][j]
                bad = np.where((q < lo - TH["r5_lim_m"]) | (q > hi + TH["r5_lim_m"]))[0]
                if len(bad):
                    out.append(_ev("R5", "joint_limit", ep, uid, S["t"][bad[0]], S["t"][bad[-1]], float(np.nanmax(np.abs(q[bad] - np.clip(q[bad], lo, hi)))),
                                   TH["r5_lim_m"], "m / rad（越限量）", "B", YELLOW, cov, joint=j, group="data_integrity_flags"))
    if not lim and S["artq"]:
        unk.append(dict(rule_id="R5", missing_fields=["header.limits"], note="关节体限位没读到，越限查不了"))
    return out, unk


# ------------------------------------------------------------------ R6
def r6(ep, S):
    out, unk = [], []
    task = (ep.get("header") or {}).get("task")
    cov = _coverage(S)
    s = S["succ"]
    if S["n"] and np.isfinite(s).any():
        closed = np.zeros(S["n"], bool)
        for g in S["grip"].values():
            closed |= g < 0.9 * 0.05
        first_close = int(np.argmax(closed)) if closed.any() else S["n"]
        hit = np.where(s > 0.5)[0]
        if len(hit) and (hit[0] == 0 or hit[0] < first_close):
            out.append(_ev("R6", "success_without_action", ep, None, S["t"][hit[0]], S["t"][hit[0]], int(hit[0]), "第 0 行或合爪前就成功", "步", "B",
                           YELLOW, cov, cause="开局白给 / 没抓放就成功（E11 一类）", official_s=1, group="criterion_flags"))
    else:
        unk.append(dict(rule_id="R6", missing_fields=["state.succ"]))
    v = VERIFIERS.get(task)
    if v is None:
        unk.append(dict(rule_id="R6", missing_fields=["replay_s（原样源码离线复算）", "verified_s（独立语义）"], note=f"{task} 没有复算插件（VERIFIERS）"))
    return out, unk


# ------------------------------------------------------------------ R7（跨集）
def r7(eps):
    out = []
    groups = defaultdict(list)
    for ep in eps:
        h = ep.get("header") or {}
        if h.get("seed") is None:
            continue
        ver = ((h.get("versions") or {}).get("official") or {}).get("sha")
        groups[(h.get("task"), h.get("seed"), ver, json.dumps(h.get("config_sha"), sort_keys=True))].append(ep)
    for key, g in groups.items():
        if len(g) < 2:
            continue
        s0 = (g[0]["steps"][0] if g[0]["steps"] else {}).get("state") or {}
        for ep in g[1:]:
            s1 = (ep["steps"][0] if ep["steps"] else {}).get("state") or {}
            worst = dict(pos=0.0, rot=0.0, q=0.0)
            for k in s0:
                if k.endswith("_x") and (k.startswith("obj_") or k.startswith("art_")) and k in s1:
                    p = k[:-2]
                    try:
                        d = math.dist([s0[f"{p}_{a}"] for a in "xyz"], [s1[f"{p}_{a}"] for a in "xyz"])
                        r = rot_deg(np.array([s0[f"{p}_{a}"] for a in AXES[3:]], float), np.array([s1[f"{p}_{a}"] for a in AXES[3:]], float))
                    except (KeyError, TypeError):
                        continue
                    if s0.get(f"{p}_z", 0) is not None and s0.get(f"{p}_z", 0) < TH["hidden_z"]:
                        continue
                    worst["pos"], worst["rot"] = max(worst["pos"], d), max(worst["rot"], r if r == r else 0)
                if k.startswith("qpos_") and k in s1:
                    try:
                        worst["q"] = max(worst["q"], abs(float(s0[k]) - float(s1[k])))
                    except (TypeError, ValueError):
                        pass
            if worst["pos"] > TH["r7_pos"] or worst["rot"] > TH["r7_rot_deg"] or worst["q"] > TH["r7_q"]:
                out.append(_ev("R7", "pair_mismatch", ep, None, 0, 0, {k: round(v, 6) for k, v in worst.items()},
                               dict(pos_m=TH["r7_pos"], rot_deg=TH["r7_rot_deg"], q_rad=TH["r7_q"]), "m / deg / rad", "B", YELLOW, "—",
                               cause=f"同种子 {key[1]} 和第 {g[0]['header'].get('episode')} 集开局不同（不能当严格配对）", group="opening_flags"))
    return out


# ------------------------------------------------------------------ R8
def r8(ep, S, appearance="interval"):
    out = []
    calls = []
    for s in S["rows"]:
        for c in s.get("rand") or []:
            if c.get("cls") in APPEAR_CLS or c.get("method") in APPEAR_METH:
                calls.append((int(s["t"]), c.get("functor"), c.get("cls"), c.get("method")))
    ts = sorted({c[0] for c in calls})
    if appearance == "reset-only" and ts:
        out.append(_ev("R8", "mid_episode_change", ep, None, ts[0], ts[-1], len(ts), "reset-only 配置下集内不该变", "次", "A（earlier measurements）", RED, "—",
                       cause="外观事件集内触发（配置说 reset 即随机）", steps=ts[:50], group="appearance_flags"))
    elif ts:
        d = np.diff([0] + ts)
        pattern = bool(len(ts) >= 2 and all(x % 10 == 0 for x in ts))
        if pattern or ts[0] > 1:
            out.append(_ev("R8", "appearance_schedule", ep, None, ts[0], ts[-1], dict(first=ts[0], n=len(ts), stride10=pattern),
                           "首次触发 > 第 1 步（前段默认灯）或步距全是 10 的倍数", "步", "A（earlier measurements）", YELLOW, "—",
                           cause="interval 外观事件：时钟捷径风险（门六同源）", steps=ts[:50], strides=sorted(set(int(x) for x in d))[:10],
                           group="appearance_flags"))
    return out


# ------------------------------------------------------------------ 开局检查（十任务；出 opening_flags.csv 给协议 / 漏斗第 7 节）
TOOLS_ROOT = os.environ.get("RSREC_TOOLS_ROOT", "")   # optional external tooling; absent => task-specific opening checks are skipped
OPEN_TH = dict(success_s=0.5, slide_s=0.8, slide_m=0.005, far_m=0.25, art_m=0.005)
GENERIC_OPEN = ("开局即成功", "落定前滑开", "关节体不在初值", "白给")


def _tools_checks():
    """Optional external task-specific opening checks (drawer, sample loading). None available => ({}, None)."""
    try:
        if not TOOLS_ROOT:
            return {}, None
        if TOOLS_ROOT not in sys.path:
            sys.path.insert(0, TOOLS_ROOT)
        from evaltool_draft import opening as OP
        from evaltool_draft.adapters.robosyn import OPENING_CHECKS
        return OPENING_CHECKS, OP
    except Exception:  # noqa: BLE001
        return {}, None


def _cols(ep):
    rows = [s for s in ep["steps"] if s.get("state") is not None]
    keys = set().union(*[set(s["state"]) for s in rows]) if rows else set()
    return {k: np.array([_num(s["state"].get(k)) for s in rows], float) for k in keys}


def _freebie(ep, task):
    """→ (0 / 1 / None, note). Uses an optional external `judges` module; missing or task unknown => None."""
    try:
        fin = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if fin not in sys.path:
            sys.path.insert(0, fin)
        from judges import SPECS, judge_rows
    except Exception:  # noqa: BLE001
        return None, None
    if task not in SPECS:
        return None, None
    rows, cur = [], {}
    for s_ in ep["steps"]:
        st = s_.get("state")
        if st is None:
            continue
        cur.update(st)
        rows.append(dict(cur))
    if not rows:
        return None, None
    h = ep.get("header") or {}
    sc = (((h.get("opening") or {}).get("readback") or {}).get("scales")) or {}
    try:
        v = judge_rows(task, rows, dict(rigid=sc), float(h.get("rate_hz") or 25.0))
    except Exception:  # noqa: BLE001
        return None, None
    if v.get("reason") == "缺列":
        return None, None
    # 开局检查只看开局那一刻（「整集没搬动就判成」是整集的事，留在严判据 reason=白给 里，不进开局旗）
    hit = bool(v.get("freebie_start"))
    return int(hit), ("开局（第 0 行）官方成功或缩放后任务条件已成立" if hit else None)


def opening_refs(eps):
    """一批里的开局参考：关节体每个关节开局值的中位（≥ 3 集才用）、物体开局四元数中位（tilt 用）。"""
    q0 = defaultdict(list)
    quats = defaultdict(list)
    for ep in eps:
        st = (ep["steps"][0].get("state") or {}) if ep["steps"] else {}
        for k, v in st.items():
            if k.startswith("art_") and "_qpos_" in k and isinstance(v, (int, float)):
                q0[k].append(float(v))
            if k.endswith("_qw") and k.startswith("obj_"):
                p = k[:-3]
                try:
                    quats[p].append([float(st[f"{p}_q{a}"]) for a in "wxyz"])
                except (KeyError, TypeError, ValueError):
                    pass
    art = {k: float(np.median(v)) for k, v in q0.items() if len(v) >= 3}
    qref = {}
    for p, Q in quats.items():
        Q = np.asarray(Q, float)
        Q = Q * np.sign(Q[:, 0:1] + 1e-12)
        qref[p] = np.median(Q, 0)
    return dict(art=art, quat=qref, n=len(eps))


def opening_check(ep, refs=None):
    """→ (flags {检查名: 0 / 1 / None, any}, 证据 {检查名: 说明})。十任务都查三项通用；抽屉 / 上样另加外部专项（可选）。"""
    S = series(ep)
    h = ep.get("header") or {}
    task = h.get("task")
    fl, ev = {}, {}
    dt = float(np.median(_dts(S))) if S["n"] > 1 else S["dt0"]
    # 1 开局即成功：头 0.5 s 内官方成功
    if S["n"] and np.isfinite(S["succ"]).any():
        n = min(S["n"], int(round(OPEN_TH["success_s"] / dt)) + 1)
        hit = np.where(S["succ"][:n] > 0.5)[0]
        fl["开局即成功"] = int(len(hit) > 0)
        if len(hit):
            ev["开局即成功"] = f"第 {int(S['t'][hit[0]])} 行官方成功"
    else:
        fl["开局即成功"] = None
    # 2 落定前滑开：头 0.8 s 双法兰都离物体 > 0.25 m 时，任务物体（不含干扰物、隐藏池）离开局 > 5 mm
    n = min(S["n"], int(round(OPEN_TH["slide_s"] / dt)) + 1)
    objs = [u for u, e in S["ent"].items() if e["kind"] == "obj" and _visible(S, u) and not u.startswith("distractor")]
    if objs and n > 1:
        worst = None
        for u in objs:
            a = S["ent"][u]["a"][:n, :3]
            far = _eef_dist(S, S["ent"][u]["a"][:, :3])[:n] > OPEN_TH["far_m"]
            mv = np.linalg.norm(a - a[0], axis=1)
            m = float(np.nanmax(np.where(far, mv, 0.0)))
            if worst is None or m > worst[1]:
                worst = (u, m)
        fl["落定前滑开"] = int(worst[1] > OPEN_TH["slide_m"])
        if fl["落定前滑开"]:
            ev["落定前滑开"] = f"{worst[0]} 头 {OPEN_TH['slide_s']} s 动了 {worst[1] * 1000:.1f} mm（机器人远）"
    else:
        fl["落定前滑开"] = None
    # 3 关节体不在初值：参考 = 这一批开局中位（≥ 3 集）；否则 = 回合头限位里离 q0 近的那一端
    lim = h.get("limits") or {}
    st0 = (ep["steps"][0].get("state") or {}) if ep["steps"] else {}
    arts = [k for k in st0 if k.startswith("art_") and "_qpos_" in k]
    if arts:
        bad = []
        for k in arts:
            v = _num(st0.get(k))
            ref = (refs or {}).get("art", {}).get(k)
            src = "批中位"
            if ref is None:
                uid, j = k[4:].rsplit("_qpos_", 1)
                L = lim.get(uid)
                if L and int(j) < len(L["lo"]):
                    lo, hi = L["lo"][int(j)], L["hi"][int(j)]
                    ref, src = (lo if abs(v - lo) <= abs(v - hi) else hi), "近端限位"
                else:
                    ref, src = 0.0, "假定 0"
            thr = OPEN_TH["art_m"]
            uid_, j_ = k[4:].rsplit("_qpos_", 1)
            L_ = lim.get(uid_)
            if L_ and int(j_) < len(L_["lo"]):          # 行程短的（铃盖 5 mm）：阈值 = min(5 mm, 行程 × 0.3)
                thr = min(thr, 0.3 * abs(L_["hi"][int(j_)] - L_["lo"][int(j_)]))
            if np.isfinite(v) and abs(v - ref) > thr:
                bad.append(f"{k} = {v:.4f}（参考 {ref:.4f}，{src}；阈值 {thr:.4f}）")
        fl["关节体不在初值"] = int(bool(bad))
        if bad:
            ev["关节体不在初值"] = "；".join(bad[:4])
    else:
        fl["关节体不在初值"] = None
    # 4 freebie: a strict-judge freebie = 开局（第 0 行）官方 succ / 官方原样重算 / 缩放后任务条件就成立。
    #   例：table_rearrangement seed 541083718（both variants succeed at the start）。和「开局即成功」（头 0.5 s 官方成功）的差别：还看缩放后的任务条件
    fl["白给"], why = _freebie(ep, task)
    if why:
        ev["白给"] = why
    # task-specific (optional external tooling: drawer, sample loading)
    checks, OP = _tools_checks()
    spec = checks.get(task) or []
    counted = set(GENERIC_OPEN)
    if spec and OP is not None:
        c = _cols(ep)
        prog = ((ep.get("footer") or {}).get("progress") or {})
        x = dict(rings=prog.get("rings")) if prog.get("rings") else None
        qrefs = {chk["obj"]: (refs or {}).get("quat", {}).get(chk["obj"]) for chk in spec if chk.get("kind") == "tilt"}
        for chk in spec:
            name = f"{task}:{chk['name']}"
            try:
                if chk.get("kind") == "art_pre_move" and x is None:
                    v = None
                elif chk.get("kind") == "tilt" and qrefs.get(chk["obj"]) is None:
                    v = None
                else:
                    v = OP.check(c, chk, x, 1.0 / dt, qrefs)
            except Exception as e:  # noqa: BLE001
                v, ev[name] = None, f"出错 {type(e).__name__}: {e}"[:120]
            fl[name] = v
            if chk.get("counts", True):
                counted.add(name)
    fl["any"] = int(any(fl.get(k) == 1 for k in counted))
    return fl, ev


OPEN_CSV_VERSION = "gate7-open/1"


def write_opening_csv(path, eps, items=None):
    """opening_flags.csv, columns: tool_version, task, seed, configured, any, open_<检查名>…
    （0 / 1 / 空 = 判不了）。configured = 1：十任务都配了三项通用检查。种子为空的集不写。"""
    import csv
    refs = opening_refs(eps)
    rows, names = [], []
    for ep in eps:
        fl, _ = opening_check(ep, refs)
        seed = (ep.get("header") or {}).get("seed")
        if seed is None:
            continue
        rows.append((int(seed), fl))
        names += [k for k in fl if k != "any" and k not in names]
    tasks = {int((ep.get("header") or {}).get("seed")): (ep.get("header") or {}).get("task") for ep in eps
             if (ep.get("header") or {}).get("seed") is not None}
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tool_version", "task", "seed", "configured", "any"] + [f"open_{k}" for k in names])
        for seed, fl in rows:
            w.writerow([OPEN_CSV_VERSION, tasks.get(seed), seed, 1, fl["any"]] + ["" if fl.get(k) is None else fl[k] for k in names])
    return len(rows), names


# ------------------------------------------------------------------ 合起来
def check_episode(ep, appearance="interval"):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _check_episode(ep, appearance)


def _check_episode(ep, appearance="interval"):
    S = series(ep)
    evs, unk = [], []
    evs += r0(ep, S)
    evs += r1_r3(ep, S)
    evs += r2(ep, S)
    e4, u4 = r4(ep, S)
    e5, u5 = r5(ep, S)
    e6, u6 = r6(ep, S)
    evs += e4 + e5 + e6 + r8(ep, S, appearance)
    unk += u4 + u5 + u6
    if S["contacts"] is None:
        unk.append(dict(rule_id="R1/R3", missing_fields=["step.contacts"], note="没有接触列：R1 / R3 只能黄色候选（不能确认环境自运动）"))
    return evs, unk


def _group(e):
    if e.get("group"):
        return e["group"]
    return dict(R0="data_integrity_flags", R5="data_integrity_flags", R6="criterion_flags", R8="appearance_flags").get(e["rule_id"], "mid_episode_flags")


def gate_env(eps, appearance="interval", strict=False, stop_frac=None, opening_csv=None):
    stop_frac = TH["stop_frac"] if stop_frac is None else stop_frac
    items, cnt = [], defaultdict(int)
    per_ep = {}
    orefs = opening_refs(eps)
    for ep in eps:
        evs, unk = check_episode(ep, appearance)
        try:
            fl, why = opening_check(ep, orefs)
        except Exception as e:  # noqa: BLE001
            fl, why = {"any": None}, {"出错": repr(e)[:160]}
        for k, v in fl.items():
            if k != "any" and v == 1:
                evs.append(_ev("O", k, ep, None, 0, 0, why.get(k), "见 OPEN_TH / 外部专项", "—", "B", YELLOW, "—",
                               cause="开局检查", group="opening_flags"))
        unk += [dict(rule_id="O", sub=k, missing_fields=["判不了（缺列或缺参考）"]) for k, v in fl.items() if k != "any" and v is None]
        ep["_opening"] = fl
        per_ep[id(ep)] = (evs, unk)
    for e in r7(eps):
        for ep in eps:
            if (ep.get("header") or {}).get("episode") == e["episode"] and (ep.get("header") or {}).get("seed") == e["seed"]:
                per_ep[id(ep)][0].append(e)
                break
    n_obj = 0
    for ep in eps:
        evs, unk = per_ep[id(ep)]
        h = ep.get("header") or {}
        n_obj += len([k for k in ((ep["steps"][0].get("state") or {}) if ep["steps"] else {}) if k.endswith("_qw")])
        grp = {g: [] for g in GROUPS}
        for e in evs:
            if strict and e["level"] == YELLOW:
                e = dict(e, level=RED, note="strict：黄当红")
            grp[_group(e)].append(e)
            cnt[f"{e['rule_id']}.{e['sub']}"] += 1
        lv = RED if any(e["level"] == RED for g in grp.values() for e in g) else (
            YELLOW if any(e["level"] == YELLOW for g in grp.values() for e in g) else GREEN)
        items.append(dict(episode=h.get("episode"), seed=h.get("seed"), path=ep.get("path"), 级=lv, unknown=unk, opening=ep.get("_opening"), **grp))
    lvls = [it["级"] for it in items]
    n_red = lvls.count(RED)
    lv = RED if n_red else (YELLOW if YELLOW in lvls else GREEN)
    frac = n_red / len(items) if items else 0.0
    n_csv = None
    if opening_csv:
        n_csv = write_opening_csv(opening_csv, eps)
    return {"门": "七 · 环境异常", "级": lv, "opening_flags_csv": dict(path=opening_csv, 行数=n_csv[0], 列=n_csv[1]) if n_csv else None, "集数": len(items), "查了对象数": n_obj, "红集数": n_red, "黄集数": lvls.count(YELLOW),
            "丢弃率": round(frac, 4), "停批线": stop_frac, "停批": frac > stop_frac, "事件计数": dict(cnt),
            "说明": "B 级规则只出黄色候选；红 = 数据坏（R0）/ 开局抽屉已开（R2 A 级）/ reset-only 下集内外观变（R8）；--strict 黄当红；注 = 机器人近时的突跳 / 掉桌，只记不算",
            "明细": items}
