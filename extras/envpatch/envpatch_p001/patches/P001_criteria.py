# -*- coding: utf-8 -*-
"""P001 · 搅拌开关「推到开位并停住」的判据读法（纯函数，不 import 仿真）。

判据读法：θ ≥ 0.8 θmax 停 ≥ 0.5 s 且左手推；见 README。

    switched_on：θ ≥ 0.8 θmax 连续 ≥ 0.5 s，且这 0.5 s 里左手指和拨杆没有接触（片自己停在开位，
                 不是被手按着）。满足一次就锁住。
    flip_by_left：θ 第一次过 θmid 之前 0.5 s 内，有 switch_lever × left_link7 / left_link8 的接触，
                  冲量 ≥ 0.01。排除被烧杯、机身或右手碰过去的情况。

输入都是逐步序列（一步 = 一个控制步，dt = 1 / env.control_freq）：
    theta     每步 θ（rad，0 = 关位）
    contact   每步「左手指 × 拨杆」是否有接触（bool）
    impulse   每步该接触的冲量（float，没接触给 0）

读数列：flip_frac_end / flip_frac_max / flip_hold_s / flip_by_left。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

ON_FRAC = 0.8
HOLD_S = 0.5
LEFT_WINDOW_S = 0.5
IMPULSE_MIN = 0.01


def _as_list(x) -> list:
    return [float(v) for v in x]


def switched_on(theta: Sequence[float], contact: Sequence[bool], dt: float, theta_max: float,
                frac: float = ON_FRAC, hold_s: float = HOLD_S) -> dict:
    """返回 {ok, hold_s, first_step}：ok 是锁存后的结果，hold_s 是最长一次「开位且不碰手」连续时长。"""
    if len(theta) != len(contact):
        raise ValueError(f"theta 与 contact 长度不一致：{len(theta)} vs {len(contact)}")
    if theta_max <= 0:
        raise ValueError("theta_max 必须 > 0")
    need = max(1, int(math.ceil(hold_s / dt - 1e-9)))
    run = 0
    best = 0
    first = None
    for i, (t, c) in enumerate(zip(theta, contact)):
        if t >= frac * theta_max and not c:
            run += 1
            if run > best:
                best = run
            if run >= need and first is None:
                first = i - need + 1
        else:
            run = 0
    return {"ok": first is not None, "hold_s": round(best * dt, 4),
            "first_step": first, "need_steps": need}


def flip_by_left(theta: Sequence[float], contact: Sequence[bool], impulse: Sequence[float],
                 dt: float, theta_max: float, window_s: float = LEFT_WINDOW_S,
                 impulse_min: float = IMPULSE_MIN) -> dict:
    """第一次过 θmid 之前 window_s 内，是否有「左手指 × 拨杆」且冲量 ≥ impulse_min 的接触。"""
    mid = 0.5 * theta_max
    cross = next((i for i, t in enumerate(theta) if t > mid), None)
    if cross is None:
        return {"ok": False, "cross_step": None, "best_impulse": 0.0}
    back = max(0, cross - int(math.ceil(window_s / dt)))
    best = 0.0
    for i in range(back, cross + 1):
        if contact[i]:
            best = max(best, float(impulse[i]))
    return {"ok": best >= impulse_min, "cross_step": cross, "best_impulse": round(best, 6)}


def episode_metrics(theta: Sequence[float], contact: Sequence[bool], impulse: Sequence[float],
                    dt: float, theta_max: float, frac: float = ON_FRAC,
                    hold_s: float = HOLD_S, window_s: float = LEFT_WINDOW_S,
                    impulse_min: float = IMPULSE_MIN) -> dict:
    """一集的全部读数列（判据用，不含「烧杯不倒 ∧ 离搅拌器 ≤ 8 cm」两条官方条件）。"""
    theta = _as_list(theta)
    contact = [bool(c) for c in contact]
    impulse = _as_list(impulse)
    if not theta:
        return {"switched_on": False, "flip_by_left": False, "flip_frac_end": 0.0,
                "flip_frac_max": 0.0, "flip_hold_s": 0.0, "n_steps": 0}
    on = switched_on(theta, contact, dt, theta_max, frac, hold_s)
    by = flip_by_left(theta, contact, impulse, dt, theta_max, window_s, impulse_min)
    return {
        "switched_on": bool(on["ok"]),
        "flip_by_left": bool(by["ok"]),
        "flip_frac_end": round(theta[-1] / theta_max, 4),
        "flip_frac_max": round(max(theta) / theta_max, 4),
        "flip_hold_s": on["hold_s"],
        "flip_on_first_step": on["first_step"],
        "flip_cross_step": by["cross_step"],
        "flip_contact_impulse": by["best_impulse"],
        "n_steps": len(theta),
    }


class SwitchMonitor:
    """逐步收样本，episode 末给 episode_metrics。补丁把它挂在 env 上（env._p001_monitor）。"""

    def __init__(self, dt: float, theta_max: float, **kw):
        self.dt = float(dt)
        self.theta_max = float(theta_max)
        self.kw = dict(kw)
        self.theta: list = []
        self.contact: list = []
        self.impulse: list = []
        self.contact_error_steps: int = 0
        self.last_contact_error: str = ""

    def reset(self):
        self.theta.clear()
        self.contact.clear()
        self.impulse.clear()
        self.contact_error_steps = 0
        self.last_contact_error = ""

    def update(self, theta: float, contact: bool, impulse: float = 0.0, error: str = ""):
        self.theta.append(float(theta))
        self.contact.append(bool(contact))
        self.impulse.append(float(impulse) if contact else 0.0)
        if error:
            self.contact_error_steps += 1
            self.last_contact_error = str(error)

    def metrics(self) -> dict:
        m = episode_metrics(self.theta, self.contact, self.impulse, self.dt, self.theta_max, **self.kw)
        m["contact_error_steps"] = int(self.contact_error_steps)
        m["last_contact_error"] = self.last_contact_error
        return m


def contact_pairs(user_ids, impulse, lever_ids, left_ids) -> tuple:
    """同 contact_from_user_ids，但把冲量一起算。返回 (是否有, 最大冲量)。"""
    import numpy as np

    def _np(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x)

    u = _np(user_ids)
    imp = _np(impulse).reshape(-1)
    u = u.reshape(-1, 2)
    if u.shape[0] == 0 or imp.size == 0:
        return False, 0.0
    lever = [int(x) for x in np.asarray(lever_ids).reshape(-1)]
    left = [int(x) for x in np.asarray(left_ids).reshape(-1)]
    a, b = u[:, 0], u[:, 1]
    pair = ((np.isin(a, lever) & np.isin(b, left)) | (np.isin(b, lever) & np.isin(a, left)))
    if not pair.any():
        return False, 0.0
    return True, float(np.max(imp[pair]))
