"""mixer_operating: step-wise FAQ criteria.

FAQ: (1) pick up the beaker on the right, (2) put it stably at the correct place on the mixer (on the small circle with the cross),
(3) press the button on the left side of the mixer.

The simulator's built-in check (``mixer_operating.py``): beaker not fallen (axis < 60 deg from vertical) and beaker within 8 cm
(horizontally) of the mixer origin and the button region touched at some point.  The small circle is tighter than 8 cm and flipping the
switch is stricter than touching, so the criteria below follow the FAQ:

| FAQ sentence          | check                                                                                                         | value |
| beaker lifted         | a step where a right-hand link touches the beaker and the beaker touches neither the table nor the mixer      |       |
| on the circle         | the beaker's bottom centre is within the small circle's radius (parts: beaker_mixer.ring) of its centre,      |       |
|                       | measured in the circle plane                                                                                  |       |
| resting stably        | set once the beaker was seen touching the mixer (while touching neither the table nor the robot) and cleared   | 60 deg|
|                       | whenever it touches the table, touches the robot or tips over (axis >= 60 deg from vertical)                  |       |
| button pressed        | the switch joint angle reached >= 0.8 x its full range (parts: mixer_switch.flip.range) and, before that, a    | 0.8   |
|                       | left-hand link had touched the switch lever                                                                   |       |
| switch on, hand off   | on this step the switch angle is still >= 0.8 x range and no left-hand link touches the lever                 | 0.8   |
| order                 | lifted < placed < pressed                                                                                     |       |
| hold                  | all of the above for >= 20 consecutive steps up to the last step                                              | 20    |

Note: the official mixer asset has no switch joint (the button is a contact region).  Without a ``mixer_switch`` articulation
(range missing from the parts table, or joint not readable) the two switch sentences and the order check are reported as ``None``
(not enough evidence), so the episode can never be counted as FAQ-successful.  The switch-based sentences need an asset that
models the switch as a joint (``mixer_switch``, link ``switch_lever``).
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "mixer_operating"
UIDS = ("beaker", "beaker_mixer", "mixer_switch")
RIGID = ("beaker", "beaker_mixer")
ARTS = ("mixer_switch",)    # optional: skipped if the scene has no such articulation
ART_LINKS = {}
EEF = ()
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
ON_FRAC = 0.8
FALL_DEG = 60.0
LEVER = "mixer_switch:switch_lever"
DEFINITION = ("beaker lifted = right hand touches the beaker and it touches neither table nor mixer; on the circle = beaker bottom centre within the small circle's radius of its centre; "
              "resting stably = beaker seen on the mixer (not touching table or robot), cleared when it touches the table or robot or tips over (axis >= 60 deg); "
              "button pressed = switch angle reached >= 0.8 x full range and a left-hand link had touched the lever before; switch on, hand off = angle still >= 0.8 x range and no left-hand "
              "contact with the lever; order = lifted < placed < pressed; hold = all of them for >= 20 consecutive steps up to the last step. "
              "Without a switch joint the switch sentences are reported as None.")


class Judge:
    def __init__(self, parts):
        parts = P.parts_of(parts)
        self.ring, self.bottom = parts["beaker_mixer"]["ring"], parts["beaker"]["bottom"]
        self.r_small = float(self.ring["section"]["r_out"])
        rng = (parts.get("mixer_switch") or {}).get("flip", {}).get("range")
        self.has_switch = rng is not None
        self.q_lo, self.q_hi = (float(rng[0]), float(rng[1])) if self.has_switch else (0.0, 0.0)
        self.resting = False
        self.t_pick = self.t_place = self.t_press = None
        self.lever_seen = False
        self.press_ok = False
        self.tail = C.Tail()
        self.i = -1

    def step(self, obs):
        """obs: T_mixer, T_beaker (4x4), q_switch (switch joint angle, nan / None if unavailable), t, and five contact flags (True / False):
        beaker_table, beaker_mixer, beaker_robot, beaker_right (right-hand link touches the beaker), lever_left (left-hand link touches the lever)."""
        self.i += 1
        t = int(obs.get("t", self.i))
        Tm, Tb = np.asarray(obs["T_mixer"], float), np.asarray(obs["T_beaker"], float)
        c = C.world_point(Tm, self.ring["p"])
        n = C.world_dir(Tm, self.ring["normal"])
        d = C.world_point(Tb, self.bottom["p"]) - c
        g = float(d @ n)
        radial = float(np.linalg.norm(d - g * n))
        on_circle = radial <= self.r_small
        fall = C.tilt_deg(Tb[:3, 2]) >= FALL_DEG
        on_table, on_mixer, rob = bool(obs["beaker_table"]), bool(obs["beaker_mixer"]), bool(obs["beaker_robot"])
        held = bool(obs["beaker_right"]) and not on_table and not on_mixer
        if on_mixer and not on_table and not rob:
            self.resting = True
        if on_table or rob or fall:
            self.resting = False
        lever = bool(obs["lever_left"])
        self.lever_seen |= lever
        q = obs.get("q_switch")
        q = float("nan") if q is None else float(q)
        sw_known = self.has_switch and bool(np.isfinite(q))
        sw = bool(sw_known and self.q_hi > self.q_lo and (q - self.q_lo) >= ON_FRAC * (self.q_hi - self.q_lo))
        if held and self.t_pick is None:
            self.t_pick = t
        if self.t_pick is not None and self.t_place is None and self.resting and t > self.t_pick:
            self.t_place = t
        if sw and self.t_press is None:
            self.t_press, self.press_ok = t, self.lever_seen
        order = bool(self.t_pick is not None and self.t_place is not None and self.t_press is not None and self.t_pick < self.t_place < self.t_press)
        conds = {"beaker_lifted": self.t_pick is not None, "on_circle": bool(on_circle), "resting_stably": bool(self.resting),
                 "button_pressed": bool(self.t_press is not None and self.press_ok) if sw_known or self.t_press is not None else None,
                 "switch_on_hand_off": bool(sw and not lever) if sw_known else None,
                 "order": order if sw_known or self.t_press is not None else None}
        done = C.tri(conds) is True
        k = self.tail.push(done, t)
        return dict(done=done, hold_n=k, conds=conds,
                    m=dict(beaker_bottom_to_circle_centre_mm=radial * 1e3, circle_radius_mm=self.r_small * 1e3, beaker_bottom_over_circle_mm=g * 1e3, switch_angle_rad=q,
                           switch_full_range_rad=self.q_hi, t_lifted=self.t_pick, t_placed=self.t_place, t_pressed=self.t_press))


def make_judge(ctx):
    return Judge(ctx.parts)


def obs_from_scene(scene, ctx=None):
    ps = scene.pair_set()
    tb = lambda pred: any(a == "beaker" and pred(b) for a, b in ps)
    return dict(T_mixer=scene.obj_T("beaker_mixer"), T_beaker=scene.obj_T("beaker"), q_switch=scene.art_qpos("mixer_switch", 0, float("nan")),
                beaker_table=tb(lambda u: u == "table"), beaker_mixer=tb(lambda u: u == "beaker_mixer"), beaker_robot=tb(lambda u: u.startswith("robot:")),
                beaker_right=tb(lambda u: u.startswith("robot:right_")),
                lever_left=any(a == LEVER and b.startswith("robot:left_") for a, b in ps), t=scene.t)
