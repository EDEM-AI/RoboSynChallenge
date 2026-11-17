"""One detector per automatically checkable known environment bug (rule ids E01-E32, local to this tool) plus three extra non-E examples.

结论五种（verdict）：
  confirmed  环境 / 判据 / 记录的错，证据够强（A 级阈值，或 B 级阈值 + 接触列证明机器人没碰）⇒ 按 action 处置
  candidate  像环境错，但缺强证据（多半是没有接触列）⇒ 先抽核，不自动排除
  unknown    拿不准（距离在灰带里、缺字段、缺参考）⇒ 不排除，列出缺什么
  expert     现象有，但是专家 / 策略引起的（碰过、早焊、动作表短）⇒ 算专家失败，修专家；**不当环境错放过**
  clear      查了，没有；na = 这条不适用本任务 / 本数据
kind：env_physics（物理自运动 / 穿模 / 飞出）· env_reset（开局 / 重置）· criterion（判据）· recording（记录 / 进程）· appearance（外观事件）· interface（接口单位）· expert
action（造数据处置）：exclude（训练视图排除、原数据留）· spot_check（抽核）· official_only（标签册：官方成、严不成）· expert_fail（专家失败，进修专家清单）· flag（只打标签，不排除）· none

每个结果：{det, bugs, verdict, kind, action, t0, t1, values, rule, missing, why}；t0 / t1 是录制步号 t（官方证据 = step − 1）。
Threshold grades: A = existing code / official criterion / earlier evidence, B = engineering suggestion (candidate only unless the contact column gives strong evidence).
"""
from __future__ import annotations

import math

import numpy as np

from .ctx import Ctx, quat_R

RULESET = "envanom/0.1"
TASK_BUGS = {   # 判据类分歧按任务落到哪条 E
    "sample_loading": "E06/E07", "manipulate_pipette": "E08", "drawer_open_place": "E10", "table_rearrangement": "E11",
    "items_handover": "E12/E13", "click_bell": "X1", "handle_basket": "X2", "item_assembly": "E09", "mixer_operating": "E14",
    "water_pouring": "E14"}
CRIT_REASONS = {"条件没成立", "白给", "官方时长门槛没过"}           # 判官 reason：判据 / 几何的分歧
EXPERT_REASONS = {"没松手", "松手后没静止", "没静止", "集尾不够 N 步", "没连续 N 步"}   # 判官 reason：做法质量（专家 / 策略的事）

TH = dict(
    e01_gap=0.005, e01_move=0.005, e01_self=0.02, e01_at_handle=0.05, e01_far=0.10, e01_reach=0.03, pre_s=0.5,
    e03_xy=0.20, e03_dz=0.03,
    e04_xy=0.06,
    e02_rot_deg=2.0, e02_z=0.005, e02_min_batch=5,
    e09_lat=0.02, e09_rel_m=0.002, e09_rel_deg=1.0, e09_hold_s=1.0, e09_carry_m=0.03,
    x1_depth=0.0048, x1_plateau_m=1e-4, x1_plateau_s=0.4,
    x2_count=55, x2_max_steps=500,
    e18_dq=0.03,
    e21_touch=0.05, e21_far=0.25, e21_far_c=0.10, e21_size=0.05,
)
DEFAULT_ACTION = dict(confirmed="exclude", candidate="spot_check", unknown="flag", expert="expert_fail", clear="none", na="none")


def R(det, bugs, verdict, kind=None, action=None, t0=None, t1=None, values=None, rule=None, missing=None, why=None):
    return dict(det=det, bugs=bugs, verdict=verdict, kind=kind, action=action or DEFAULT_ACTION[verdict],
                t0=None if t0 is None else int(t0), t1=None if t1 is None else int(t1), values=values or {},
                rule=f"{RULESET}:{rule or det}", missing=missing or [], why=why)


def _r(x, k=4):
    try:
        return round(float(x), k)
    except (TypeError, ValueError):
        return None


def _first(mask):
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None


# ================================================================== E01 抽屉留缝 / 自开
def d_e01(c: Ctx, batch=None):
    if c.task != "drawer_open_place":
        return [R("E01", "E01", "na")]
    if not c.has("art_drawer_qpos_0"):
        return [R("E01", "E01", "unknown", missing=["state.art_drawer_qpos_0"])]
    q = c.col("art_drawer_qpos_0")
    lim = ((c.h.get("limits") or {}).get("drawer") or {}).get("lo")
    q_closed = float(lim[0]) if lim else 0.0
    out = []
    gap = float(q[0] - q_closed)
    if gap > TH["e01_gap"]:
        out.append(R("E01.gap", "E01", "confirmed", "env_reset", "flag", 0, 0, dict(q0_m=_r(gap), thr=TH["e01_gap"]), "E01.gap/A",
                     why="开局抽屉已开（官方 issue #80 留缝，A 级）；开局检查单列、不扣分母，造数据先推回（v1.3 预防段），不排除"))
    # 把手位置：有把手连杆用连杆；没有 ⇒ 柜根 ⊗ (0.105 + q, 0, 0.035)（用一批录制拟合，残差 < 1 mm）
    Hp = c.pose("artlink_drawer_handle_xpos")
    if Hp is not None:
        H, hsrc = Hp[:, :3], "把手连杆"
    else:
        P = c.pose("art_drawer")
        if P is None:
            out.append(R("E01.self_open", "E01", "unknown", missing=["state.art_drawer_<pose>"]))
            return out
        H = np.array([P[i, :3] + quat_R(P[i, 3:7]) @ np.array([0.105 + (q[i] if np.isfinite(q[i]) else 0.0), 0.0, 0.035]) for i in range(c.n)])
        hsrc = "柜根 + 把手偏移（代理）"
    d = c.tip_dist(H)
    _, tsrc = c.tips()
    dq = q - q[0]
    on = _first(dq > TH["e01_move"])
    if on is None:
        out.append(R("E01.self_open", "E01", "clear", values=dict(dq_max_m=_r(np.nanmax(dq)))))
        return out
    reach = _first(d < TH["e01_reach"])
    w = c.steps(TH["pre_s"])
    dmin = float(np.nanmin(d[max(0, on - w):on + 1]))
    end = reach if (reach is not None and reach > on) else c.n
    dq_pre = float(np.nanmax(dq[on:end])) if end > on else float(dq[on])
    vals = dict(onset_t=int(c.t[on]), dq_before_reach_m=_r(dq_pre), tip_handle_min_m=_r(dmin), reach_t=None if reach is None else int(c.t[reach]),
                handle=hsrc, hand=tsrc)
    touch, tt = c.robot_touch("drawer", on - w, on)
    if dmin <= TH["e01_at_handle"]:
        out.append(R("E01.self_open", "E01", "clear", values=vals, why="开度起动那一刻手在把手上（正常拉开）"))
    elif touch:
        vals["robot_contact_t"] = int(c.t[tt])
        out.append(R("E01.self_open", "E01", "expert", "expert", t0=c.t[on], t1=c.t[min(end, c.n) - 1], values=vals,
                     why="手不在把手，但接触列里机器人（或手里的东西）碰了抽屉 ⇒ 专家碰开，不是自开"))
    elif dmin > TH["e01_far"]:
        strong = dq_pre >= TH["e01_self"]
        if touch is False and strong:
            out.append(R("E01.self_open", "E01", "confirmed", "env_physics", t0=c.t[on], t1=c.t[min(end, c.n) - 1], values=vals, rule="E01.self_open/A+contact",
                         why=f"手到把手前抽屉自己开了 {dq_pre * 100:.1f} cm（≥ 2 cm，A 级），手离把手 > 10 cm，接触列里机器人没碰抽屉"))
        else:
            miss = [] if touch is not None else ["step.contacts（这集没有接触列）"]
            out.append(R("E01.self_open", "E01", "candidate", "env_physics", t0=c.t[on], t1=c.t[min(end, c.n) - 1], values=vals, missing=miss,
                         rule="E01.self_open/A" if strong else "E01.creep/B",
                         why=("手到把手前抽屉自己开了 ≥ 2 cm、手离把手 > 10 cm" if strong else "手到把手前抽屉动了 5–20 mm（B 级，只抽核）")
                         + ("；没有接触列，排除不了前臂 / 手里物体碰柜" if touch is None else "")))
    else:
        out.append(R("E01.self_open", "E01", "unknown", t0=c.t[on], values=vals, why="起动时手离把手 5–10 cm：灰带，人看"))
    return out


# ================================================================== E03 叉勺生在柜内 / 上方
def d_e03(c: Ctx, batch=None):
    if c.task != "drawer_open_place":
        return [R("E03", "E03", "na")]
    P = c.pose("art_drawer")
    if P is None or c.n == 0:
        return [R("E03", "E03", "unknown", missing=["state.art_drawer_<pose>"])]
    ds = sorted({k[4:-2] for k in (c.rows[0].get("state") or {}) if k.startswith("obj_distractor") and k.endswith("_x")})
    vis = []
    for u in ds:
        a = c.pose(f"obj_{u}")
        if a is None or not np.isfinite(a[0, 2]) or a[0, 2] < -1.0:
            continue
        vis.append((u, a[0]))
    if not vis:
        return [R("E03", "E03", "clear", values=dict(n_distractors=0))]
    zs = np.array([v[1][2] for v in vis])
    tab = ((c.h.get("geom") or {}).get("table") or {}).get("top_z")
    z_ref = float(tab) if tab is not None else float(np.median(zs))
    hits = []
    for u, p in vis:
        dxy = float(np.hypot(p[0] - P[0, 0], p[1] - P[0, 1]))
        if dxy < TH["e03_xy"] and p[2] - z_ref > TH["e03_dz"]:
            hits.append(dict(uid=u, xy_to_cabinet_m=_r(dxy), above_ref_m=_r(p[2] - z_ref)))
    if hits:
        return [R("E03", "E03", "candidate", "env_reset", t0=0, t1=0, values=dict(hits=hits, z_ref="桌面" if tab is not None else "干扰物开局 z 中位"),
                  rule="E03/B", why="开局干扰物在柜心 0.2 m 内且高出桌面 > 3 cm（旧筛查辅助，B）；柜体碰撞几何没记，确认不了")]
    return [R("E03", "E03", "clear", values=dict(n_distractors=len(vis)))]


# ================================================================== E04 搅拌干扰物和目标重叠
def d_e04(c: Ctx, batch=None):
    if c.task != "mixer_operating":
        return [R("E04", "E04", "na")]
    if c.n == 0:
        return [R("E04", "E04", "unknown", missing=["step"])]
    st = c.rows[0].get("state") or {}
    tg = [u for u in ("beaker", "beaker_mixer") if f"obj_{u}_x" in st]
    ds = sorted({k[4:-2] for k in st if k.startswith("obj_distractor") and k.endswith("_x")})
    hits = []
    for u in ds:
        try:
            p = np.array([st[f"obj_{u}_{a}"] for a in "xyz"], float)
        except (KeyError, TypeError):
            continue
        if not np.isfinite(p).all() or p[2] < -1.0:
            continue
        for t in tg:
            q = np.array([st[f"obj_{t}_{a}"] for a in "xyz"], float)
            dxy = float(np.hypot(*(p[:2] - q[:2])))
            if dxy < TH["e04_xy"]:
                hits.append(dict(uid=u, target=t, xy_m=_r(dxy)))
    if not ds:
        return [R("E04", "E04", "clear", values=dict(n_distractors=0))]
    if hits:
        return [R("E04", "E04", "candidate", "env_reset", t0=0, t1=0, values=dict(hits=hits), rule="E04/B",
                  why="开局干扰物离烧杯 / 搅拌器 < 6 cm（1.5 m 避让满足不了时采样退化）；没有碰撞几何，只抽核")]
    return [R("E04", "E04", "clear", values=dict(n_distractors=len(ds)))]


# ================================================================== E02 组装右杆开局滚转遗留（要一批的参考）
def opening_ref_e02(ctxs):
    qs, zs = [], []
    for c in ctxs:
        if c.task != "item_assembly" or c.n == 0:
            continue
        P = c.pose("obj_guijiao2")
        if P is None or not np.isfinite(P[0]).all():
            continue
        q = P[0, 3:7] * np.sign(P[0, 3] + 1e-12)
        qs.append(q)
        zs.append(P[0, 2])
    if len(qs) < TH["e02_min_batch"]:
        return None
    return dict(q=np.median(np.array(qs), 0), z=float(np.median(zs)), n=len(qs))


def d_e02(c: Ctx, batch=None):
    if c.task != "item_assembly":
        return [R("E02", "E02", "na")]
    ref = (batch or {}).get("e02")
    P = c.pose("obj_guijiao2")
    if P is None:
        return [R("E02", "E02", "unknown", missing=["state.obj_guijiao2_<pose>"])]
    if ref is None:
        return [R("E02", "E02", "unknown", missing=[f"同批开局参考（组装集 < {TH['e02_min_batch']}）"])]
    rot = c.G.rot_deg(ref["q"], P[0, 3:7])
    dz = abs(float(P[0, 2]) - ref["z"])
    vals = dict(rot_dev_deg=_r(rot, 3), z_dev_m=_r(dz), ref_n=ref["n"])
    if rot > TH["e02_rot_deg"] or dz > TH["e02_z"]:
        return [R("E02", "E02", "candidate", "env_reset", t0=0, t1=0, values=vals, rule="E02/B",
                  why="开局右杆姿态离同批中位 > 2° 或 z > 5 mm（random 配置写死 rot / z，偏了多半是上一集遗留）；v2.1 若随机化了杆姿态这条要换参考")]
    return [R("E02", "E02", "clear", values=vals)]


# ================================================================== 判据类：E06 / E07 / E08 / E10 / E11 / E13 ＋ 白给（需要外部 judges 模块）
def d_crit(c: Ctx, batch=None):
    v = c.judge()
    bugs = TASK_BUGS.get(c.task, "—")
    if v is None:
        return [R("CRIT", bugs, "unknown", missing=["external judges module not importable, or task not one of the ten"])]
    if v.get("error"):
        return [R("CRIT", bugs, "unknown", missing=[f"judges 出错：{v['error']}"])]
    if v.get("reason") == "缺列":
        return [R("CRIT", bugs, "unknown", missing=["judges 缺列"] + list(v.get("missing") or [])[:5])]
    vals = dict(official=v.get("official"), strict=v.get("strict"), bucket=v.get("bucket"), reason=v.get("reason"),
                freebie_start=v.get("freebie_start"), freebie_unmoved=v.get("freebie_unmoved"), scales_src=(v.get("scales") or {}).get("src"))
    out = []
    if v.get("freebie_start"):
        out.append(R("CRIT.freebie", "E11/白给", "confirmed", "criterion", "exclude", 0, 0, vals, "CRIT.freebie/A",
                     why="开局（第 0 行）官方成功或缩放后任务条件已成立（judges 0.2 白给）：不当示范，评测分母照留"))
    b, why = v.get("bucket"), v.get("reason")
    if b == "official_only":
        by_expert = any(((r.get("src") or {}).get("act")) == "专家" for r in c.rows[:5])
        if why in ("集尾不够 N 步", "没松手") and not by_expert:
            out.append(R("CRIT.official_only", bugs, "unknown", None, "flag", values=vals, rule="CRIT.official_only/judges0.2",
                         why=f"评测（官方 / 策略自评）在成功那一步就停（{why}）：成功后的松手 / 0.4 s 看不到，截断造成，不归执行方也不归判据；要分请用续走（judges --after-success）"))
        elif why in CRIT_REASONS:
            out.append(R("CRIT.official_only", bugs, "candidate", "criterion", "official_only", values=vals, rule="CRIT.official_only/judges0.2",
                         why=f"官方成、严判据不成（{why}）：判据 / 缩放几何的分歧，进 official_only 册、label_valid=0；严判据阈值 B 级没独立验收，抽核"))
        else:
            out.append(R("CRIT.official_only", bugs, "expert", "expert", "official_only", values=vals, rule="CRIT.official_only/judges0.2",
                         why=f"官方成、严判据不成（{why}）：执行方（{'专家' if by_expert else '策略'}）的做法质量（没放稳 / 没持续），不算环境"))
    if not out:
        out.append(R("CRIT", bugs, "clear", values=vals))
    return out


# ================================================================== X1 按铃行程够不着官方 4.8 mm（0.9 桶）
def d_x1(c: Ctx, batch=None):
    if c.task != "click_bell":
        return [R("X1", "X1", "na")]
    if not c.has("art_button_qpos_0"):
        return [R("X1", "X1", "unknown", missing=["state.art_button_qpos_0"])]
    q = c.col("art_button_qpos_0")
    press = -q
    lim = (c.h.get("limits") or {}).get("button")
    succ = bool(np.nanmax(c.S["succ"]) > 0.5) if np.isfinite(c.S["succ"]).any() else None
    if lim and lim.get("lo") and lim.get("hi"):
        stroke = abs(float(lim["hi"][0]) - float(lim["lo"][0]))
        vals = dict(stroke_m=_r(stroke, 5), thr=TH["x1_depth"], press_max_m=_r(np.nanmax(press), 5))
        if stroke < TH["x1_depth"]:
            return [R("X1", "X1", "confirmed", "criterion", "flag", values=vals, rule="X1/A",
                      why="铃盖行程（header.limits）< 官方写死的 4.8 mm：按到底也判不成；不算专家失败，这一桶只进测试（v1.3）")]
        return [R("X1", "X1", "clear", values=vals)]
    pm = float(np.nanmax(press)) if np.isfinite(press).any() else np.nan
    w = c.steps(TH["x1_plateau_s"])
    plateau = int(np.sum(np.abs(press - pm) < TH["x1_plateau_m"])) >= w
    vals = dict(press_max_m=_r(pm, 5), plateau_steps=int(np.sum(np.abs(press - pm) < TH["x1_plateau_m"])), thr=TH["x1_depth"])
    if succ is False and plateau and 0.003 < pm < TH["x1_depth"]:
        return [R("X1", "X1", "candidate", "criterion", "spot_check", values=vals, missing=["header.limits.button"], rule="X1/B",
                  why="按到一个平台（≥ 0.4 s 不再变深）却 < 4.8 mm、官方没成：像行程够不着；没有限位读回，确认不了")]
    return [R("X1", "X1", "clear", values=vals)]


# ================================================================== X2 提篮集尾截断（专家动作表尾巴短，不是环境）
def d_x2(c: Ctx, batch=None):
    if c.task != "handle_basket":
        return [R("X2", "X2", "na")]
    if not c.has("m_in_basket"):
        return [R("X2", "X2", "unknown", missing=["state.m_in_basket"])]
    inb = c.col("m_in_basket") > 0.5
    if c.has("m_moved_left"):
        mv = c.col("m_moved_left") > 0.5
    elif c.has("m_basket_y_disp"):
        mv = (c.col("m_basket_y_disp") > 0.15) & inb
    else:
        return [R("X2", "X2", "unknown", missing=["state.m_moved_left / m_basket_y_disp"])]
    goal = inb & mv
    succ = bool(np.nanmax(c.S["succ"]) > 0.5) if np.isfinite(c.S["succ"]).any() else None
    run = 0
    for g in goal[::-1]:
        if not g:
            break
        run += 1
    vals = dict(goal_at_end=bool(goal[-1]) if len(goal) else None, end_count=run, need=TH["x2_count"], n_steps=c.n, official=succ)
    if succ is False and len(goal) and goal[-1] and run < TH["x2_count"]:
        if c.n < TH["x2_max_steps"]:
            return [R("X2", "X2", "expert", "expert", t0=c.t[-run], t1=c.t[-1], values=vals, rule="X2/A",
                      why=f"集尾奶在篮里且左移达标，计数只到 {run} < 55 就结束（集长 {c.n} < 500）：动作表尾巴短，专家补停顿，不是环境")]
        return [R("X2", "X2", "expert", "expert", t0=c.t[-run], t1=c.t[-1], values=vals, rule="X2/A", why="到 500 步上限计数还不够 55：做法太晚，不是环境")]
    return [R("X2", "X2", "clear", values=vals)]


# ================================================================== E09 组装早焊锁死（算专家失败）
def _rel(P1, P2):
    out = np.full((len(P1), 7), np.nan)
    for i in range(len(P1)):
        R1 = quat_R(P1[i, 3:7])
        if not np.isfinite(R1).all():
            continue
        out[i, :3] = R1.T @ (P2[i, :3] - P1[i, :3])
        q1, q2 = P1[i, 3:7] / np.linalg.norm(P1[i, 3:7]), P2[i, 3:7] / np.linalg.norm(P2[i, 3:7])
        w1, v1 = q1[0], -q1[1:]
        w2, v2 = q2[0], q2[1:]
        out[i, 3] = w1 * w2 - v1 @ v2
        out[i, 4:] = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return out


def d_e09(c: Ctx, batch=None):
    if c.task != "item_assembly":
        return [R("E09", "E09", "na")]
    P1, P2 = c.pose("obj_guijiao1"), c.pose("obj_guijiao2")
    if P1 is None or P2 is None:
        return [R("E09", "E09", "unknown", missing=["state.obj_guijiao1/2_<pose>"])]
    rel = _rel(P1, P2)
    hold = c.steps(TH["e09_hold_s"])
    n = c.n
    # 焊上 = 从某步起到集尾，管 2 在管 1 系里不动（位 < 2 mm、角 < 1°），且这段里两件一起被搬动过 > 3 cm（排除都躺在桌上）
    w = None
    last = rel[-1]
    ok = np.array([np.linalg.norm(rel[i, :3] - last[:3]) < TH["e09_rel_m"] and c.G.rot_deg(rel[i, 3:7], last[3:7]) < TH["e09_rel_deg"]
                   if np.isfinite(rel[i]).all() else False for i in range(n)])
    if ok.all():
        w = 0
    else:
        bad = np.where(~ok)[0]
        w = int(bad[-1]) + 1 if bad[-1] + 1 < n else None
    if w is None or n - w < hold:
        return [R("E09", "E09", "clear", values=dict(weld="没见到锁定段"))]
    carry = float(np.nanmax(np.linalg.norm(P1[w:, :3] - P1[w, :3], axis=1)))
    if w == 0 or carry < TH["e09_carry_m"]:
        return [R("E09", "E09", "clear", values=dict(weld="相对不动但没一起搬动（多半都在桌上）", carry_m=_r(carry)))]
    # 焊那一刻的横向错位（官方原式：管 2 心到管 1 x 轴的距离）
    R1 = quat_R(P1[w, 3:7])
    u = R1[:, 0] / np.linalg.norm(R1[:, 0])
    diff = P2[w, :3] - P1[w, :3]
    lat = float(np.linalg.norm(diff - u * (diff @ u)))
    ang = float(np.degrees(math.acos(min(1.0, abs(float(R1[:, 0] @ quat_R(P2[w, 3:7])[:, 0]))))))
    vals = dict(weld_t=int(c.t[w]), lateral_at_weld_m=_r(lat), angle_at_weld_deg=_r(ang, 2), carry_after_m=_r(carry), thr=TH["e09_lat"])
    if lat > TH["e09_lat"]:
        return [R("E09", "E09", "expert", "expert", t0=c.t[w], t1=c.t[-1], values=vals, rule="E09/A",
                  why="焊那一刻横向 > 2 cm，之后锁死：锁死是环境规矩，但焊前没对准是专家的事 ⇒ S7 没达成 · 子类「早焊锁死」")]
    return [R("E09", "E09", "clear", values=vals)]


# ================================================================== E12 / E31 多环境
def d_e12(c: Ctx, batch=None):
    ne = c.h.get("num_envs")
    if ne is None:
        return [R("E12/E31", "E12/E31", "unknown", missing=["header.num_envs"])]
    if int(ne) > 1:
        return [R("E12/E31", "E12/E31", "candidate", "criterion", values=dict(num_envs=ne), why="num_envs > 1：递笔 env0 广播 / 登记索引配对风险，按 env 重算")]
    return [R("E12/E31", "E12/E31", "na", values=dict(num_envs=ne), why="num_envs = 1，不触发")]


# ================================================================== E17 外观事件（门七 R8）
def d_e17(c: Ctx, batch=None):
    if c.ep.get("source") == "official":
        return [R("E17", "E17", "na", missing=["step.rand"], why="官方证据没记 setter 调用，查不了")]
    ev = c.G.r8(c.ep, c.S)
    if ev:
        e = ev[0]
        return [R("E17", "E17", "candidate", "appearance", "flag", e["t_start"], e["t_end"], dict(observed=e["observed"]), "E17/R8",
                  why="interval 外观事件集内触发（时钟捷径风险，门六管）；不是物理异常，不排除")]
    return [R("E17", "E17", "clear")]


# ================================================================== E18 关节体崩
def d_e18(c: Ctx, batch=None):
    out = []
    for uid, js in c.S["artq"].items():
        for j, q in js.items():
            bad = np.where(~np.isfinite(q))[0]
            dq = np.abs(np.diff(q))
            big = np.where(dq > TH["e18_dq"])[0]
            if len(bad):
                out.append(R("E18", "E18", "confirmed", "env_physics", c.t[bad[0]], c.t[bad[-1]], dict(uid=uid, joint=j, n_nonfinite=int(len(bad))), "E18/A",
                             why="关节读数非有限：关节体崩"))
            elif len(big):
                i = int(big[0]) + 1
                touch, _ = c.robot_touch(uid, i - 2, i)
                vals = dict(uid=uid, joint=j, dq_step_m=_r(dq[big[0]]), thr=TH["e18_dq"])
                if touch:
                    out.append(R("E18", "E18", "expert", "expert", c.t[i], c.t[i], vals, why="关节一步跳 > 3 cm，同时机器人在碰：撞出来的"))
                else:
                    out.append(R("E18", "E18", "candidate", "env_physics", c.t[i], c.t[i], vals, "E18/B",
                                 missing=[] if touch is not None else ["step.contacts"], why="关节一步跳 > 3 cm（25 fps 下 > 0.75 m/s）"))
    return out or [R("E18", "E18", "clear" if c.S["artq"] else "na")]


# ================================================================== E19 / E30 记录字段无效
def d_e19(c: Ctx, batch=None):
    out = []
    for e in c.G.r0(c.ep, c.S):
        if e["sub"] == "quat_norm" and e.get("cause"):
            out.append(R("E19", "E19", "confirmed", "recording", "flag", e["t_start"], e["t_end"], dict(uid=e["uid"]), "E19/A",
                         why="全 0 位姿（关节体当刚体读）：这一列记 unknown，不能据它判自运动；不排除这一集"))
        elif e["sub"] in ("nonfinite", "t_not_increasing", "sim_t_not_increasing"):
            out.append(R("R0", "R0", "confirmed", "recording", "exclude", e["t_start"], e["t_end"], dict(uid=e["uid"], sub=e["sub"], n=e["observed"]), "R0/B",
                         why="记录坏（非有限 / 时间不递增）"))
    return out or [R("E19", "E19", "clear")]


def d_e30(c: Ctx, batch=None):
    V = c.S["vel"]
    if V is None:
        return [R("E30", "E30", "na", why="这集没有逐步速度（rich 子集才记）")]
    zero_move = []
    for uid, v in V.items():
        if uid not in c.S["ent"]:
            continue
        a = c.S["ent"][uid]["a"]
        sp = np.r_[0, np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1) / c.dt]
        moving = sp > 0.05
        if moving.sum() >= 5:
            vv = np.linalg.norm(v[:, :3], axis=1)
            frac0 = float(np.mean(vv[moving] < 1e-6))
            same = float(np.mean(np.abs(np.linalg.norm(v[:, 3:6], axis=1) - vv)[moving] < 1e-9))
            if frac0 > 0.9:
                zero_move.append(dict(uid=uid, why="动着时速度读回全 0", frac=_r(frac0, 3)))
            elif same > 0.9:
                zero_move.append(dict(uid=uid, why="角速度 = 线速度（E30 CPU 组 getter）", frac=_r(same, 3)))
    if zero_move:
        return [R("E30", "E30", "confirmed", "recording", "flag", values=dict(bad=zero_move[:6]), rule="E30/A",
                  why="速度列读回不可用（物体在动、getter 给 0 或角速度读成线速度）：R5 动态自洽、速度门一律不用这列，改用位姿差分；不排除这一集")]
    return [R("E30", "E30", "clear")]


# ================================================================== E21 物理异常：自运动 / 单步跳 / 速度突跳 / 穿桌 / 掉桌（门七 R1 R3 R4 再分因）
def _row_of(c, t):
    if t is None:
        return None
    i = int(np.searchsorted(c.t, t))
    return min(max(i, 0), c.n - 1)


def d_e21(c: Ctx, batch=None):
    out = []
    try:
        evs = c.G.r1_r3(c.ep, c.S)
    except ValueError as e:      # gate7 r1_r3 raises on an empty window when a single-step jump sits on row 1 (known issue in gate7; handled here)
        evs = []
        out.append(R("E21", "E21", "unknown", missing=[f"门七 r1_r3 出错：{e}"[:120]], why="门七 R1/R3 在这集算不了（第 1 行单步跳 ⇒ 空窗），R4 照算"))
    e4, _ = c.G.r4(c.ep, c.S)
    evs += e4
    seen_fall = set()
    w = c.steps(TH["pre_s"])
    for e in evs:
        sub, uid = e["sub"], e["uid"]
        if uid is None or uid not in c.S["ent"]:
            continue
        i0, i1 = _row_of(c, e["t_start"]), _row_of(c, e["t_end"])
        lo = max(0, i0 - w)
        a = c.S["ent"][uid]["a"]
        if e["rule_id"] == "R3" and i0 >= 3 and np.isfinite(a[i0 - 3:i0 + 1, 2]).all() and a[i0 - 3, 2] - a[i0, 2] > 0.1:
            # 落地冲击：前几步在快速下落，这一步停住（撞地 / 撞桌）——R3 的「突跳」是落地，不是物理 bug。回溯到开始下落那一步再分因
            z = a[:, 2]
            f = i0 - 3
            while f > 0 and np.isfinite(z[f - 1]) and z[f - 1] - z[f] > 0.002:
                f -= 1
            d = c.tip_dist(a[:, :3])[max(0, f - w):f + 2]
            dmin = float(np.nanmin(d)) if len(d) else float("inf")
            touch, _ = c.robot_touch(uid, f - w, f + 1)
            key = (uid, f)
            if key in seen_fall:
                continue
            seen_fall.add(key)
            vals = dict(rule="R3→落地", uid=uid, fall_start_t=int(c.t[f]), drop_m=_r(z[f] - z[i0]), tip_min_m=_r(dmin), contacts=c.has_contacts)
            if touch or dmin <= TH["e21_touch"]:
                out.append(R("E21.fall", "E21", "expert", "expert", "none", c.t[f], e["t_end"], vals, "E21.fall",
                             why="开始下落前后指尖 ≤ 5 cm / 有接触：执行方碰掉桌，不是环境"))
            elif dmin > TH["e21_far"]:
                out.append(R("E21.fall", "E21", "candidate", "env_physics", None, c.t[f], e["t_end"], vals, "E21.fall/B",
                             missing=[] if touch is not None else ["step.contacts"],
                             why="开始下落时指尖 > 25 cm：没人碰却掉桌（可能被别的物体推、或从边上滑落），抽核"))
            else:
                out.append(R("E21.fall", "E21", "unknown", None, None, c.t[f], e["t_end"], vals, "E21.fall", why="开始下落时指尖 5–25 cm：人看"))
            continue
        d = c.tip_dist(a[:, :3])[lo:i1 + 1]
        dmin = float(np.nanmin(d)) if len(d) else float("inf")
        touch, tt = c.robot_touch(uid, lo, i1)
        vals = dict(rule=f"{e['rule_id']}.{sub}", uid=uid, observed=e["observed"], tip_min_m=_r(dmin), contacts=c.has_contacts)
        if sub == "settling":
            out.append(R("E21.settling", "E21/R1", "candidate", "env_reset", "flag", e["t_start"], e["t_end"], vals, "E21.settling/R1",
                         why="开局 0.5 s 内还在动（没落定就开录）：开局检查单列，不排除"))
            continue
        if touch or dmin <= TH["e21_touch"]:
            out.append(R("E21", "E21", "expert", "expert", "none", e["t_start"], e["t_end"], vals, f"E21.{sub}",
                         why="窗内机器人（或手里的东西）碰过 / 指尖 ≤ 5 cm：碰出来的，归接触 / 策略，不算环境"))
        elif touch is False and dmin > TH["e21_far_c"]:
            out.append(R("E21", "E21", "confirmed", "env_physics", None, e["t_start"], e["t_end"], vals, f"E21.{sub}/contact",
                         why="接触列里窗内（含前 0.5 s）机器人和手里的东西都没碰、指尖 > 10 cm，物体自己动 / 跳 / 穿桌"))
        elif touch is None and dmin > TH["e21_far"]:
            out.append(R("E21", "E21", "candidate", "env_physics", None, e["t_start"], e["t_end"], vals, f"E21.{sub}/B", missing=["step.contacts"],
                         why="指尖 > 25 cm 时物体自己动 / 跳 / 穿桌；没有接触列，排除不了被别的物体撞"))
        else:
            out.append(R("E21", "E21", "unknown", None, None, e["t_start"], e["t_end"], vals, f"E21.{sub}",
                         missing=[] if c.has_contacts else ["step.contacts"], why="指尖在灰带（5–25 cm，或有接触列时 5–10 cm）：人看"))
    return out or [R("E21", "E21", "clear")]


# ================================================================== E22 / E28 进程 / 记录完整性
def d_e22(c: Ctx, batch=None):
    if c.ep.get("source") == "official":
        return [R("E22/E28", "E22/E28", "na", why="官方证据按 result json 的集数读，完整性由官方产出保证")]
    ft = c.ep.get("footer")
    if c.ep.get("partial") or ft is None:
        return [R("E22/E28", "E22/E28", "confirmed", "recording", "exclude", values=dict(partial=bool(c.ep.get("partial")), footer=ft is not None, n_rows=c.n),
                  rule="E22/A", why="回合没收尾（.part / 没 footer）：进程死在中途（E28 段错误 / E22 close 先于写），这集不进训练")]
    nf = ft.get("n_steps")
    if nf is not None and abs(int(nf) - c.n) > 1:
        return [R("E22/E28", "E22/E28", "unknown", values=dict(footer_n=nf, rows=c.n), why="footer 步数和实际行数对不上")]
    return [R("E22/E28", "E22/E28", "clear")]


# ================================================================== E23 夹爪单位
def d_e23(c: Ctx, batch=None):
    g = [r.get("grip") for r in c.rows if r.get("grip")]
    if not g:
        return [R("E23", "E23", "na", missing=["step.grip（rsrec 1.2 才有）"], why="没有夹爪三列，查不了")]
    raw = np.array([x["raw"] for x in g], float)
    eff = np.array([x["eff"] for x in g], float)
    clipped = (raw > 0.0505) & (np.abs(eff - 0.05) < 1e-6)
    if clipped.any():
        return [R("E23", "E23", "confirmed", "interface", "flag", values=dict(raw_max=_r(np.nanmax(raw)), eff_max=_r(np.nanmax(eff)), frac_clipped=_r(clipped.mean(), 3)),
                  rule="E23/A", why="下发的是 0–1 语义、控制器按米截到 0.05：动作列不能按列名直接混训，用 grip.eff / norm；不是物理 bug，不排除")]
    return [R("E23", "E23", "clear")]


# ================================================================== E32 关节驱动（只能给证据，判不了）
def d_e32(c: Ctx, batch=None):
    art = ((c.h.get("phys") or {}).get("art")) or {}
    if not art:
        return [R("E32", "E32", "na", missing=["header.phys.art"] if c.S["artq"] else [], why="没有关节驱动读回")]
    jd = {u: v.get("joint_drive") for u, v in art.items() if v.get("joint_drive")}
    return [R("E32", "E32", "na", values=dict(joint_drive=jd), missing=["header.phys.art.<uid>.drive_type（录制器没读 drive_type）"],
              why="判不了：只读回刚度 / 阻尼，读不到 drive_type；读回值附在这里，E01 自开集人看时并看")]


DETECTORS = [d_e01, d_e02, d_e03, d_e04, d_crit, d_x1, d_x2, d_e09, d_e12, d_e17, d_e18, d_e19, d_e21, d_e22, d_e23, d_e30, d_e32]

# 查不了的（不写检测器，写清缺什么）
NOT_AUTO = {
    "E05": "同种子不逐位复现：跨集比较，门七 R7 已有（run 的批级 r7）；不排除训练，只是不能当严格配对",
    "E13": "递笔判倒用局部 x：要资产功能轴定义；judges 严判据并排已覆盖「官方成 / 严不成」",
    "E14": "成功锁存口径：接口问题，录制逐步 succ 已按任务源码；不是一集的异常",
    "E15": "SmolVLA 块内不查成功：只对 SmolVLA 评测；要 policy 块边界 + 逐步 succ，现有录制没有 SmolVLA",
    "E16": "SmolVLA worker seed：推理侧，录制里没有 worker seed",
    "E20": "移液器名义位姿偏移：没有参考偏移（官方功能表没给数），不编常数",
    "E24": "顶视相机随机化全 0：配置层，跨集比 cam 外参即可，不是一集的异常",
    "E25": "专家筛选只验非空计划：批级拒绝账（attempts.jsonl），不是一集的异常",
    "E26": "SmolVLA dtype：已入 main",
    "E27": "查询 / 复位隐式推进物理：要 physics_step_index 或查询前后 sim 时间，录制器没记；门七读法也不收 query 行",
    "E29": "多卡 GPU 映射：建环境阶段，看启动日志",
}
