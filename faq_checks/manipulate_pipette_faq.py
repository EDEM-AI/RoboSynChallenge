"""manipulate_pipette: step-wise FAQ criteria.

FAQ: "Pick up the pipette, move it right above the beaker, visibly press the top button, then release the button."

| FAQ sentence       | check                                                                                                              | value         |
| picked up          | pipette root > 2 cm above its start height and the right wrist within 25 cm of the root, at some step              | 2 cm, 25 cm   |
| above beaker       | after pick-up and before the press: a step where the tip is above the beaker's mouth plane and its distance to the |               |
|                    | beaker axis is < the mouth's inner radius                                                                          |               |
| button pressed     | after "above beaker": a step where the button slide <= 0.66 x slide lower limit + 0.1 mm (the simulator's own      | 0.66, 0.1 mm  |
|                    | "at min" test) and a robot finger link touches the button link                                                     |               |
| button released    | after the press: the slide is back to within 0.5 mm of the pre-press plateau (median slide reading between "above  | 0.5 mm        |
|                    | beaker" and the press) and no robot link touches the button link any more                                          |               |
| still held         | the pipette is still in the right hand and off the table, and the tip is still inside the mouth radius             |               |
| upright            | pipette long axis (local x) < 20 deg from vertical and beaker axis (local z) < 20 deg from vertical (the           | 20 deg        |
|                    | simulator requires this on every step)                                                                             |               |
| hold               | all of the above for >= 20 consecutive steps up to the last step                                                   | 20            |
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "manipulate_pipette"
UIDS = ("pipette", "beaker1")
RIGID = ("beaker1",)
ARTS = ("pipette",)
ART_LINKS = {}
EEF = ("right",)
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
LIFT_M = 0.02
HELD_M = 0.25
PRESS_FRAC = 0.66
PRESS_TOL_M = 1e-4
REL_TOL_M = 0.0005
FALL_DEG = 20.0
BUTTON_LINK = "pipette:pipette_button_link"
TCP_RIGHT_FLANGE = np.array([0.0, 0.0, 0.12])     # grasp point of the right gripper in the wrist-flange (link6) frame, from the URDF

DEFINITION = ("picked up = pipette root > 2 cm above its start and right wrist within 25 cm of it; above beaker = after pick-up and before the press, tip above the beaker's mouth plane "
              "and < the mouth's inner radius from its axis; button pressed = slide <= 0.66 x slide lower limit + 0.1 mm (the simulator's own test) with a robot finger touching the button link; "
              "button released = slide back to within 0.5 mm of the pre-press plateau and no robot link touches the button link; still held = pipette in the right hand, off the table, "
              "tip inside the mouth radius; upright = pipette and beaker axes < 20 deg from vertical; hold = all of them for >= 20 consecutive steps up to the last step")


def geo_from_parts(parts):
    """Episode parts table -> the geometry this module needs."""
    parts = P.parts_of(parts)
    m = parts["beaker1"]["mouth"]
    return dict(tip=np.asarray(parts["pipette"]["tip"]["p"], float), mouth=np.asarray(m["p"], float),
                mouth_n=C.unit(m["normal"]), r_in=float(m["section"]["r_in"]))


class Judge:
    def __init__(self, geo, q_lo):
        """geo = geo_from_parts(...); q_lo = lower limit of the button slide joint (m, negative)."""
        self.g, self.thr = geo, PRESS_FRAC * float(q_lo) + PRESS_TOL_M
        self.z0 = None
        self.t_pick = self.t_above = self.t_press = self.t_rel = None
        self.q_plat, self.q_pre, self.q_min = [], None, None
        self.press_touch = None
        self.tail = C.Tail()
        self.i = -1

    def step(self, obs):
        """obs: T_pipette (pipette root), T_beaker (4x4), q (button slide reading, m), p_right (right gripper grasp point), touch_button
        (a robot link touches the button link on this step: True / False / None), t (optional).  The first call is the start."""
        self.i += 1
        t = int(obs.get("t", self.i))
        Tp, Tb = np.asarray(obs["T_pipette"], float), np.asarray(obs["T_beaker"], float)
        q, touch = float(obs["q"]), obs.get("touch_button")
        if self.z0 is None:
            self.z0 = float(Tp[2, 3])
        lifted = float(Tp[2, 3]) - self.z0 > LIFT_M
        held = float(np.linalg.norm(np.asarray(obs["p_right"], float) - Tp[:3, 3])) < HELD_M
        d = C.world_point(Tp, self.g["tip"]) - C.world_point(Tb, self.g["mouth"])
        n = C.world_dir(Tb, self.g["mouth_n"])
        h = float(d @ n)
        rad = float(np.linalg.norm(d - h * n))
        over = rad < self.g["r_in"]
        pip_tilt, bk_tilt = C.tilt_deg(Tp[:3, 0]), C.tilt_deg(Tb[:3, 2])
        upright = pip_tilt < FALL_DEG and bk_tilt < FALL_DEG
        if self.t_pick is None and lifted and held:
            self.t_pick = t
        if self.t_pick is not None and self.t_press is None:
            if self.t_above is None and h > 0 and over:
                self.t_above = t
            if self.t_above is not None:
                if q <= self.thr and touch is not False:
                    self.t_press, self.press_touch = t, touch
                    self.q_pre = float(np.median(self.q_plat)) if self.q_plat else q
                elif q > self.thr:
                    self.q_plat.append(q)
        if self.t_press is not None:
            self.q_min = q if self.q_min is None else min(self.q_min, q)
        released = bool(self.t_press is not None and q >= self.q_pre - REL_TOL_M and touch is not True)
        if released and self.t_rel is None:
            self.t_rel = t
        if not released:
            self.t_rel = None
        conds = {"picked_up": self.t_pick is not None, "above_beaker": self.t_above is not None,
                 "button_pressed": (None if self.press_touch is None else True) if self.t_press is not None else False,
                 "button_released": (None if touch is None else True) if released else False,
                 "still_held_over_beaker": bool(held and lifted and over), "upright": bool(upright)}
        done = C.tri(conds) is True
        nrun = self.tail.push(done, t)
        return dict(done=done, hold_n=nrun, conds=conds,
                    m=dict(q_mm=q * 1e3, thr_mm=self.thr * 1e3, q_pre_mm=None if self.q_pre is None else self.q_pre * 1e3,
                           tip_r_mm=rad * 1e3, r_in_mm=self.g["r_in"] * 1e3, tip_h_mm=h * 1e3, pip_tilt_deg=pip_tilt, beaker_tilt_deg=bk_tilt,
                           held=held, lifted=lifted, touch_button=touch))


def make_judge(ctx):
    q_lo = (ctx.limits or {}).get("pipette_slide_lo")
    if q_lo is None:
        q_lo = P.parts_of(ctx.parts)["pipette"]["button_slide"]["range"][0]
    return Judge(geo_from_parts(ctx.parts), q_lo)


def obs_from_scene(scene, ctx=None):
    if not scene.has("art_pipette_qpos_0"):
        return None
    rows = scene.contacts
    touch = None if rows is None else any((str(x[0]) == BUTTON_LINK and str(x[1]).startswith("robot:")) or
                                          (str(x[1]) == BUTTON_LINK and str(x[0]).startswith("robot:")) for x in rows)
    return dict(T_pipette=scene.art_T("pipette"), T_beaker=scene.obj_T("beaker1"), q=scene.art_qpos("pipette", 0),
                p_right=C.world_point(scene.eef_T("right"), TCP_RIGHT_FLANGE), touch_button=touch, t=scene.t)
