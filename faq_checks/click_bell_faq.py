"""click_bell: step-wise FAQ criteria.

FAQ: press the yellow button area of the bell (an organiser note says pressing the bell body is enough).

The simulator's built-in check (``click_bell.py``) counts any detectable press depth and latches it.  The criteria
below are stricter and follow the FAQ more closely:

| FAQ sentence      | check                                                                                          | value  |
| pressed           | at some step the button depth is >= 0.96 x the actual stroke of this episode's bell (latched); |        |
|                   | the pressing hand (the hand closest to the bell on that step) and its height are recorded      | 0.96   |
| hand released     | the pressing hand's wrist is >= 2 cm above where it was at the press step and no robot link    | 2 cm   |
|                   | touches the bell (if contacts are unavailable only the lift is used)                           |        |
| base not pushed   | the bell base moved <= 5 mm from its start position                                            | 5 mm   |
| hold              | all of the above for >= 20 consecutive steps up to the last step                               | 20     |

Actual stroke = stroke in the parts table x the bell's body scale along the press axis (the simulator's joint-limit
reader does not follow body scale, so the unscaled URDF stroke is rescaled here).
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "click_bell"
UIDS = ("button",)
RIGID = ()
ARTS = ("button",)        # articulations read each step (pose + joint positions)
ART_LINKS = {}            # articulation -> links whose pose is read
EEF = ("left", "right")   # wrist flanges (link6) read each step
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
JUDGE_F = 0.96
LIFT_M = 0.020
BASE_MOVE_M = 0.005
DEFINITION = ("pressed = button depth >= 0.96 x actual stroke (= URDF stroke x body scale along the press axis; latched); hand released = pressing hand's "
              "wrist >= 2 cm above its height at the press step and no robot link touches the bell (if contacts are unavailable, only the lift is checked); "
              "base not pushed = bell base moved <= 5 mm from its start; hold = all of them for >= 20 consecutive steps up to the last step. "
              "The simulator's built-in check is not used.")


class Judge:
    def __init__(self, parts, scale):
        """parts = episode parts table (needs ``button.press``); scale = the bell's body_scale (one or three numbers)."""
        press = P.parts_of(parts)["button"]["press"]
        k = 1.0
        if press.get("range_src") == "sdk_qpos_limits":      # the engine's limit reader ignores body_scale
            if scale is None:
                raise ValueError("need the bell's body scale to get its actual stroke")
            sc = np.abs(np.asarray(scale, float).reshape(-1))
            ax = np.abs(C.unit(press.get("axis", [0.0, 0.0, 1.0])))
            k = float(sc[0]) if sc.size == 1 else float(sc[int(np.argmax(ax))])
        self.scale_k = k
        self.q_hi, self.travel = float(press["range"][1]) * k, float(press["travel"]) * k
        self.base0, self.pressed, self.press_t, self.hand, self.z_press, self.deep = None, False, None, None, None, 0.0
        self.tail = C.Tail()
        self.i = -1

    def step(self, obs):
        """obs: q (button joint reading, m), base (bell base position), hands = {"left": xyz, "right": xyz} (wrist positions),
        touch_bell (any robot link touching the bell: True / False / None), t.  The first call is taken as the start."""
        self.i += 1
        t = int(obs.get("t", self.i))
        base = np.asarray(obs["base"], float)
        if self.base0 is None:
            self.base0 = base.copy()
        depth = self.q_hi - float(obs["q"])
        self.deep = max(self.deep, depth)
        hands = {k: np.asarray(v, float) for k, v in obs["hands"].items()}
        if not self.pressed and self.travel > 0 and depth >= JUDGE_F * self.travel:
            self.pressed, self.press_t = True, t
            self.hand = min(hands, key=lambda a: float(np.linalg.norm(hands[a] - base)))
            self.z_press = float(hands[self.hand][2])
        lift = float(hands[self.hand][2]) - self.z_press if self.pressed else float("nan")
        touch = obs.get("touch_bell")
        away = bool(self.pressed and lift >= LIFT_M and touch is not True)
        move = float(np.linalg.norm(base - self.base0))
        conds = {"pressed": bool(self.pressed), "hand_released": away, "base_not_pushed": bool(move <= BASE_MOVE_M)}
        done = C.tri(conds) is True
        n = self.tail.push(done, t)
        return dict(done=done, hold_n=n, conds=conds,
                    m=dict(max_depth_mm=self.deep * 1e3, stroke_mm=self.travel * 1e3, threshold_mm=JUDGE_F * self.travel * 1e3, stroke_scale=self.scale_k,
                           press_step=self.press_t, press_hand=self.hand, hand_lift_mm=lift * 1e3, base_move_mm=move * 1e3, robot_touches_bell=touch))


def make_judge(ctx):
    return Judge(ctx.parts, ctx.body_scale.get("button", 1.0))   # articulations without a configured body_scale are unscaled


def obs_from_scene(scene, ctx=None):
    """Build ``Judge.step`` input from a scene; None if the button reading is missing on this step."""
    q = scene.art_qpos("button", 0)
    if not isinstance(q, (int, float)) or not np.isfinite(q):
        return None
    rows = scene.contacts
    touch = None if rows is None else any(len(p) >= 3 and p[2] > 0 and any(str(x).startswith("button") for x in p[:2])
                                          and any(str(x).startswith("robot:") for x in p[:2]) for p in rows)
    return dict(q=q, base=scene.art_pos("button"), hands={a: scene.eef_pos(a) for a in ("left", "right")}, touch_bell=touch, t=scene.t)
