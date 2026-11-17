"""drawer_open_place: step-wise FAQ criteria.

FAQ: "Fully open the upper right drawer. Place the tomato on the left side of the table into the open drawer."
(closing the drawer again is not required).

The simulator's built-in check requires the tomato to be within 0.10 m (horizontally) of the cabinet's outer box origin,
which effectively requires closing the drawer again; it does not match the FAQ, so it is not used here.

| FAQ sentence        | check                                                                                               | value |
| drawer fully opened | the opening reached >= 0.9 x the drawer's stroke at some step (stroke = joint travel of the episode) | 0.9   |
| tomato in cavity    | the tomato centre, in the inner-drawer frame, lies horizontally inside the cavity floor polygon's   |       |
|                     | bounding box and between the floor and floor + cavity height                                        |       |
| still open at end   | the opening on this step is >= 0.85 x the stroke (a drawer pushed back a little by a hand during   | 0.85  |
|                     | the run only gets reported, as ``pushed_back_mm``, not penalised)                                   |       |
| hand off tomato     | no robot link touches the tomato                                                                    |       |
| hold                | all of the above for >= 20 consecutive steps up to the last step                                    | 20    |
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "drawer_open_place"
UIDS = ("duck", "drawer")
RIGID = ("duck",)
ARTS = ("drawer",)
ART_LINKS = {"drawer": ("inner_box",)}
EEF = ()
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
FULL_FRAC = 0.9
OPEN_FRAC = 0.85
DEFINITION = ("drawer fully opened = opening reached >= 0.9 x the stroke at some step; tomato in cavity = tomato centre, in the inner-drawer frame, is inside the "
              "cavity floor's horizontal extent and between the floor and floor + cavity height; still open at end = opening >= 0.85 x stroke on this step; "
              "hand off tomato = no robot link touches the tomato; hold = all of them for >= 20 consecutive steps up to the last step. "
              "The simulator's built-in check (tomato within 0.10 m of the cabinet) effectively requires closing the drawer and is not used.")


class Judge:
    def __init__(self, parts):
        parts = P.parts_of(parts)
        d = parts["drawer"]
        self.travel = float(d["slide"]["travel"])
        poly = np.asarray(d["floor"]["poly"], float)
        self.lo, self.hi = poly[:, :2].min(0), poly[:, :2].max(0)
        self.z0 = float(poly[:, 2].min())
        self.z1 = self.z0 + float(d["cavity"]["size"]["height"])
        self.center = np.asarray(parts["duck"]["center"]["p"], float)
        self.qmax = 0.0
        self.tail = C.Tail()

    def step(self, obs):
        """obs: T_inner_box (inner-drawer link pose, 4x4), T_duck (tomato), q (opening, m), free_duck (hand off the tomato: True / False / None), t."""
        q = float(obs["q"])
        self.qmax = max(self.qmax, q)
        Ti = np.asarray(obs["T_inner_box"], float)
        loc = Ti[:3, :3].T @ (C.world_point(obs["T_duck"], self.center) - Ti[:3, 3])
        inside = bool(np.all(loc[:2] >= self.lo) and np.all(loc[:2] <= self.hi) and self.z0 <= loc[2] <= self.z1)
        free = obs.get("free_duck")
        conds = {"drawer_fully_opened": bool(self.qmax >= FULL_FRAC * self.travel), "tomato_in_cavity": inside,
                 "still_open_at_end": bool(q >= OPEN_FRAC * self.travel), "hand_off_tomato": free}
        done = C.tri(conds) is True
        n = self.tail.push(done, obs.get("t"))
        return dict(done=done, hold_n=n, conds=conds,
                    m=dict(opening_mm=q * 1e3, opening_frac=q / self.travel, pushed_back_mm=(self.qmax - q) * 1e3, pushed_back_frac=(self.qmax - q) / self.travel,
                           max_opening_mm=self.qmax * 1e3, max_opening_frac=self.qmax / self.travel, stroke_mm=self.travel * 1e3,
                           tomato_in_inner_frame_mm=[float(v) * 1e3 for v in loc]))


def make_judge(ctx):
    return Judge(ctx.parts)


def obs_from_scene(scene, ctx=None):
    if not scene.has("art_drawer_qpos_0"):
        return None
    tch = scene.robot_touching("duck")
    return dict(T_inner_box=scene.art_link_T("drawer", "inner_box"), T_duck=scene.obj_T("duck"), q=scene.art_qpos("drawer", 0),
                free_duck=None if tch is None else not tch, t=scene.t)
