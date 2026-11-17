"""合成正 / 负例（不起仿真）：offline labels / acceptance cases for each rule.

  python -B tests/test_envanom.py      # --write-result also writes tests/test_envanom_result.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from envanom.ctx import Ctx  # noqa: E402
from envanom.detectors import (d_crit, d_e01, d_e09, d_e19, d_e21, d_e22, d_e30, d_x1, d_x2)  # noqa: E402

Q0 = [1.0, 0.0, 0.0, 0.0]
FAR_L, FAR_R = np.array([0.2, 0.6, 1.3]), np.array([0.2, -0.6, 1.3])


def make_ep(task, T, fn, contacts=None, header=None, vel=None, partial=False, footer=True):
    """fn(i) → {名: (位置 xyz, 四元数 wxyz) 或 标量}；没给的手放远处。"""
    steps = []
    for i in range(T):
        st = {}
        d = fn(i)
        d.setdefault("link_left_eef", (FAR_L, Q0))
        d.setdefault("link_right_eef", (FAR_R, Q0))
        for k, v in d.items():
            if isinstance(v, tuple):
                p, q = v
                for a, x in zip("xyz", p):
                    st[f"{k}_{a}"] = float(x)
                for a, x in zip(("qw", "qx", "qy", "qz"), q):
                    st[f"{k}_{a}"] = float(x)
            else:
                st[k] = float(v)
        s = dict(kind="step", t=i, sim_t=i * 0.04, state=st)
        if contacts is not None:
            s["contacts"] = contacts(i)
        if vel is not None:
            s["vel"] = vel(i)
        steps.append(s)
    h = dict(kind="header", task=task, episode=0, seed=1, rate_hz=25, num_envs=1, timing=dict(dt_s=0.04), tcp=dict(tip=[0, -0.003, 0.141]))
    h.update(header or {})
    ep = dict(path="synthetic", header=h, steps=steps, footer=dict(kind="footer", success=False, n_steps=T) if footer else None,
              plans=[], source="rsrec", partial=partial)
    return ep


def verdicts(res, det=None):
    return [r["verdict"] for r in res if det is None or r["det"] == det]


# ---------------------------------------------------------------- 抽屉
CAB = np.array([0.8, -0.1, 0.9])


def drawer_fn(qf, hand=None):
    def f(i):
        q = qf(i)
        d = {"art_drawer": (CAB, Q0), "art_drawer_qpos_0": q,
             "artlink_drawer_handle_xpos": (CAB + np.array([0.105 + q, 0, 0.035]), Q0),
             "obj_duck": (np.array([0.5, 0.2, 0.8]), Q0)}
        hp = hand(i) if hand else None
        for side, far in (("left", FAR_L), ("right", FAR_R)):
            p = hp if (hp is not None and side == "right") else far
            d[f"link_{side}_f1"] = (p, Q0)
            d[f"link_{side}_f2"] = (p + np.array([0, 0.02, 0]), Q0)
            d[f"link_{side}_eef"] = (p + np.array([0, 0, 0.14]), Q0)
        return d
    return f


H_DRAWER = dict(limits=dict(drawer=dict(lo=[0.0], hi=[0.2])))


def ramp(t0, dq, T=60):
    return lambda i: 0.0 if i < t0 else min(dq, dq * (i - t0) / 10)


def case(name, cond, got):
    return (name, bool(cond), got)


def tests():
    out = []
    no_touch = lambda i: [["duck", "table", 1, 0.0, 0.0]]
    # 1 自开 3 cm、手远、接触列在且没碰 ⇒ confirmed
    ep = make_ep("drawer_open_place", 80, drawer_fn(ramp(10, 0.03)), contacts=no_touch, header=H_DRAWER)
    v = verdicts(d_e01(Ctx(ep)), "E01.self_open")
    out.append(case("E01 自开 3 cm·手远·有接触列没碰 ⇒ confirmed", v == ["confirmed"], v))
    # 2 同上没有接触列 ⇒ candidate
    ep = make_ep("drawer_open_place", 80, drawer_fn(ramp(10, 0.03)), header=H_DRAWER)
    v = verdicts(d_e01(Ctx(ep)), "E01.self_open")
    out.append(case("E01 自开 3 cm·没接触列 ⇒ candidate（不自动排除）", v == ["candidate"], v))
    # 3 手在把手上拉 ⇒ clear
    hand = lambda i: CAB + np.array([0.105 + ramp(10, 0.13)(i), 0, 0.035])
    ep = make_ep("drawer_open_place", 80, drawer_fn(ramp(10, 0.13), hand), header=H_DRAWER)
    v = verdicts(d_e01(Ctx(ep)), "E01.self_open")
    out.append(case("E01 手在把手上拉开 ⇒ clear（专家正常拉，不当自开）", v == ["clear"], v))
    # 4 手远、但接触列里机器人前臂碰了抽屉 ⇒ expert
    touch = lambda i: [["drawer:outer_box", "robot:right_link4", 2, 0.1, 0.0]] if 5 <= i <= 12 else []
    ep = make_ep("drawer_open_place", 80, drawer_fn(ramp(10, 0.03)), contacts=touch, header=H_DRAWER)
    v = verdicts(d_e01(Ctx(ep)), "E01.self_open")
    out.append(case("E01 手远但前臂碰柜 ⇒ expert（专家碰开）", v == ["expert"], v))
    # 5 手里的鸭子撞抽屉 ⇒ expert
    touch = lambda i: [["drawer:inner_box", "duck", 1, 0.1, 0.0], ["duck", "robot:left_link8", 1, 0.1, 0.0]] if 5 <= i <= 12 else []
    ep = make_ep("drawer_open_place", 80, drawer_fn(ramp(10, 0.03)), contacts=touch, header=H_DRAWER)
    v = verdicts(d_e01(Ctx(ep)), "E01.self_open")
    out.append(case("E01 手里的鸭子撞抽屉 ⇒ expert", v == ["expert"], v))
    # 6 开局 q0 = 6 mm ⇒ gap confirmed（flag，不排除）
    ep = make_ep("drawer_open_place", 40, drawer_fn(lambda i: 0.006), header=H_DRAWER)
    r = d_e01(Ctx(ep))
    v = [(x["verdict"], x["action"]) for x in r if x["det"] == "E01.gap"]
    out.append(case("E01 q0 = 6 mm ⇒ gap confirmed·flag", v == [("confirmed", "flag")], v))
    # 7 中途 6 mm 蠕动 ⇒ creep candidate
    ep = make_ep("drawer_open_place", 80, drawer_fn(ramp(30, 0.006)), header=H_DRAWER)
    r = [x for x in d_e01(Ctx(ep)) if x["det"] == "E01.self_open"]
    out.append(case("E01 中途开 6 mm ⇒ candidate（B 级 creep）", [x["verdict"] for x in r] == ["candidate"] and "creep" in r[0]["rule"], [(x["verdict"], x["rule"]) for x in r]))
    # 8 灰带 7 cm ⇒ unknown
    hand = lambda i: CAB + np.array([0.105 + 0.07, 0, 0.035])
    ep = make_ep("drawer_open_place", 80, drawer_fn(ramp(10, 0.03), hand), header=H_DRAWER)
    v = verdicts(d_e01(Ctx(ep)), "E01.self_open")
    out.append(case("E01 起动时手离把手 7 cm ⇒ unknown", v == ["unknown"], v))
    # 9 没把手连杆 ⇒ 用柜根代理，结论一样
    def f_noh(i):
        d = drawer_fn(ramp(10, 0.03))(i)
        d.pop("artlink_drawer_handle_xpos")
        for k in list(d):
            if "_f1" in k or "_f2" in k:
                d.pop(k)
        return d
    ep = make_ep("drawer_open_place", 80, f_noh, header=H_DRAWER)
    r = [x for x in d_e01(Ctx(ep)) if x["det"] == "E01.self_open"]
    out.append(case("E01 没把手 / 手指连杆 ⇒ 柜根 + 偏移代理，仍 candidate", [x["verdict"] for x in r] == ["candidate"] and "代理" in r[0]["values"]["handle"], r[0]["values"].get("handle")))

    # ---------------------------------------------------------------- 刚体自运动 / 跳 / 碰（E21）
    P0 = np.array([0.6, 0.0, 0.85])

    def obj_fn(px):
        return lambda i: {"obj_cup": (px(i), Q0), "obj_bottle": (np.array([0.3, 0.3, 0.85]), Q0)}
    move6 = lambda i: P0 + (np.array([min(0.006, 0.002 * max(0, i - 30)), 0, 0]))   # 0.12 s 内动 6 mm（规则：0.2 s 窗 > 5 mm、持续 ≥ 0.08 s）
    ep = make_ep("water_pouring", 80, obj_fn(move6))
    v = verdicts(d_e21(Ctx(ep)), "E21")
    out.append(case("E21 无接触平移 6 mm·手远·没接触列 ⇒ candidate", "candidate" in v and "confirmed" not in v, v))
    ep = make_ep("water_pouring", 80, obj_fn(move6), contacts=lambda i: [])
    v = verdicts(d_e21(Ctx(ep)), "E21")
    out.append(case("E21 同上·接触列在且空 ⇒ confirmed", "confirmed" in v, v))
    ep = make_ep("water_pouring", 80, obj_fn(move6), contacts=lambda i: [["cup", "robot:left_link4", 1, 0.1, 0.0]] if 25 <= i <= 45 else [])
    v = verdicts(d_e21(Ctx(ep)), "E21")
    out.append(case("E21 双手远但前臂接触 ⇒ expert（碰过不叫自运动）", v and all(x == "expert" for x in v), v))
    jump = lambda i: P0 + (np.array([0.48, 0, 0]) if i >= 40 else 0)
    ep = make_ep("water_pouring", 80, obj_fn(jump))
    v = verdicts(d_e21(Ctx(ep)), "E21")
    out.append(case("E21 一步 0.48 m·手远 ⇒ candidate", "candidate" in v, v))
    rest = lambda i: P0
    ep = make_ep("water_pouring", 80, obj_fn(rest))
    v = verdicts(d_e21(Ctx(ep)))
    out.append(case("E21 全程静止 ⇒ clear", v == ["clear"], v))
    # q 和 −q 交替：不是转动
    def flip(i):
        return {"obj_cup": (P0, Q0 if i % 2 == 0 else [-1.0, 0, 0, 0]), "obj_bottle": (np.array([0.3, 0.3, 0.85]), Q0)}
    ep = make_ep("water_pouring", 80, flip)
    v = verdicts(d_e21(Ctx(ep)))
    out.append(case("E21 q 与 −q 交替 ⇒ clear（等价四元数）", v == ["clear"], v))
    # 刚释放后的惯性滑动：前 0.2 s 一直在动 ⇒ 不是「原来静止」
    inertia = lambda i: P0 + np.array([0.004 * i if i < 40 else 0.16, 0, 0])
    ep = make_ep("water_pouring", 80, obj_fn(inertia))
    v = [x for x in d_e21(Ctx(ep)) if x["det"] == "E21" and x["verdict"] in ("confirmed", "candidate") and x["values"]["rule"].startswith("R1.self")]
    out.append(case("E21 一直在动（惯性）⇒ 不出 R1 自运动", not v, [x["values"]["rule"] for x in v]))

    # 掉桌落地：自由落体后撞地（R3 突跳）；开始下落时手在旁边 ⇒ expert，手远 ⇒ candidate
    def fall(i):
        if i < 40:
            return P0
        tt = (i - 40) * 0.04
        return np.array([P0[0], P0[1], max(0.02, P0[2] - 0.5 * 9.81 * tt * tt)])
    hand_near = lambda i: {"obj_cup": (fall(i), Q0), "obj_bottle": (np.array([0.3, 0.3, 0.85]), Q0),
                           "link_left_eef": (P0 + np.array([0, 0.03, 0.0]), Q0)}
    v = [x for x in d_e21(Ctx(make_ep("water_pouring", 80, hand_near))) if x["det"] == "E21.fall"]
    out.append(case("E21 掉桌落地·开始下落时手在旁边 ⇒ expert（碰掉桌）", [x["verdict"] for x in v] == ["expert"], [x["verdict"] for x in v]))
    v = [x for x in d_e21(Ctx(make_ep("water_pouring", 80, obj_fn(fall)))) if x["det"] == "E21.fall"]
    out.append(case("E21 掉桌落地·手远 ⇒ candidate（无人碰却掉桌）", [x["verdict"] for x in v] == ["candidate"], [x["verdict"] for x in v]))

    # ---------------------------------------------------------------- 组装早焊
    def asm(lat):
        def f(i):
            p1 = np.array([0.5, 0.0, 0.9]) + (np.array([0, 0, 0.002 * (i - 30)]) if i > 30 else 0)
            if i < 30:
                p2 = np.array([0.5, 0.2 - 0.006 * i, 0.9])
            else:
                p2 = p1 + np.array([0.02, lat, 0])
            return {"obj_guijiao1": (p1, Q0), "obj_guijiao2": (p2, Q0)}
        return f
    v = d_e09(Ctx(make_ep("item_assembly", 100, asm(0.03))))
    out.append(case("E09 焊那一刻横向 3 cm ⇒ expert（早焊锁死算专家失败）", verdicts(v) == ["expert"], [(x["verdict"], x["values"].get("lateral_at_weld_m")) for x in v]))
    v = d_e09(Ctx(make_ep("item_assembly", 100, asm(0.005))))
    out.append(case("E09 横向 5 mm 焊 ⇒ clear", verdicts(v) == ["clear"], [(x["verdict"], x["values"].get("lateral_at_weld_m")) for x in v]))

    # ---------------------------------------------------------------- 提篮集尾截断
    def basket(i):
        return {"obj_milk": (np.array([0.5, 0.2, 0.9]), Q0), "obj_basket": (np.array([0.5, 0.2, 0.85]), Q0),
                "m_in_basket": 1.0 if i >= 10 else 0.0, "m_moved_left": 1.0 if i >= 287 else 0.0, "succ": 0.0}
    v = d_x2(Ctx(make_ep("handle_basket", 335, basket)))
    out.append(case("X2 集尾计数 48 < 55、集长 335 ⇒ expert（专家补停顿，不算环境）", verdicts(v) == ["expert"] and v[0]["values"]["end_count"] == 48, v[0]["values"]))

    # ---------------------------------------------------------------- 按铃行程
    bell = lambda i: {"art_button": (np.array([0.6, 0, 0.85]), Q0), "art_button_qpos_0": -min(0.0045, 0.0003 * i), "succ": 0.0}
    v = d_x1(Ctx(make_ep("click_bell", 60, bell, header=dict(limits=dict(button=dict(lo=[-0.0045], hi=[0.0]))))))
    out.append(case("X1 铃盖行程 4.5 mm < 4.8 ⇒ confirmed（判据够不着，不是专家）", verdicts(v) == ["confirmed"], v[0]["values"]))
    v = d_x1(Ctx(make_ep("click_bell", 60, bell, header=dict(limits=dict(button=dict(lo=[-0.0055], hi=[0.0]))))))
    out.append(case("X1 行程 5.5 mm ⇒ clear", verdicts(v) == ["clear"], v[0]["values"]))
    v = d_x1(Ctx(make_ep("click_bell", 60, bell)))
    out.append(case("X1 没限位、按到 4.5 mm 平台没成 ⇒ candidate", verdicts(v) == ["candidate"], v[0]["values"]))

    # ---------------------------------------------------------------- 记录类
    ep = make_ep("water_pouring", 30, obj_fn(rest), partial=True, footer=False)
    v = d_e22(Ctx(ep))
    out.append(case("E22/E28 .part 没 footer ⇒ confirmed·exclude", [(x["verdict"], x["action"]) for x in v] == [("confirmed", "exclude")], verdicts(v)))
    zero = lambda i: {"obj_cup": (P0, Q0), "art_drawer": (np.zeros(3), [0.0, 0, 0, 0])}
    v = d_e19(Ctx(make_ep("drawer_open_place", 30, zero)))
    out.append(case("E19 关节体全 0 位姿 ⇒ confirmed·flag（列无效、集不丢）", ("confirmed", "flag") in [(x["verdict"], x["action"]) for x in v], verdicts(v)))
    mv = lambda i: P0 + np.array([0.004 * i, 0, 0])
    ep = make_ep("water_pouring", 40, obj_fn(mv), vel=lambda i: {"cup": [0, 0, 0, 0, 0, 0]})
    v = d_e30(Ctx(ep))
    out.append(case("E30 物体在动、速度读回全 0 ⇒ confirmed·flag", verdicts(v) == ["confirmed"], verdicts(v)))
    ep = make_ep("water_pouring", 40, obj_fn(mv), vel=lambda i: {"cup": [0.1, 0, 0, 0, 0, 0]})
    out.append(case("E30 速度读回正常 ⇒ clear", verdicts(d_e30(Ctx(ep))) == ["clear"], None))
    # 缺接触列不能写「无接触」：E21 没接触列时不出 confirmed（上面 candidate 一例已证）；判官导不进 ⇒ unknown
    v = d_crit(Ctx(make_ep("not_a_task", 10, obj_fn(rest))))
    out.append(case("CRIT 任务不在十任务 ⇒ unknown", verdicts(v) == ["unknown"], verdicts(v)))
    return out


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    res = tests()
    lines = [f"{'过' if ok else '没过'} · {n} · {g}" for n, ok, g in res]
    n_ok = sum(ok for _, ok, _ in res)
    lines.append(f"共 {len(res)} 项，过 {n_ok}，没过 {len(res) - n_ok}")
    txt = "\n".join(lines)
    print(txt)
    if "--write-result" in sys.argv:
        open(os.path.join(HERE, "test_envanom_result.txt"), "w", encoding="utf-8").write(txt + "\n")
    sys.exit(0 if n_ok == len(res) else 1)
