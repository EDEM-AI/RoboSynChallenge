"""table_rearrangement: step-wise FAQ criteria.

FAQ: put the spoon and the fork stably on the two sides of the plate, roughly parallel, neatly arranged.

The simulator's built-in check (``table_rearrangement.py``): spoon at plate centre y - 16 cm, fork at plate centre y + 16 cm (each within 2 cm), and
neither more than 5 cm above the plate.  16 cm is hard-coded for the default plate; with a different plate diameter "the two sides" has to follow
the plate, so the criteria below use this episode's geometry:

| FAQ sentence    | check                                                                                                              | value                  |
| on both sides   | fork and spoon centres are on opposite sides of the plate centre in world y, both farther from the plate centre    |                        |
|                 | (horizontally) than the plate radius                                                                               |                        |
| roughly parallel| the long axes (local x) of fork and spoon make <= 15 deg in the table plane                                       | 15 deg                 |
| placed stably   | both were touched by a robot link at some point (really carried); no robot link touches either on this step; each  | 5 mm, 10 deg,          |
|                 | is back to its start flat height +- 5 mm with its long axis < 10 deg from horizontal; per-step displacement        | 0.1 mm, 0.1 deg        |
|                 | <= 0.1 mm and long-axis turn <= 0.1 deg                                                                            |                        |
| hold            | all of the above for >= 20 consecutive steps up to the last step                                                   | 20                     |

"Neatly arranged" is not checked separately; it is the three sentences together.
Geometry: object centres are taken as the centre of each object's local bounding box (episode body scale applied); the plate radius is half the
longest edge of the plate's local bounding box.
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "table_rearrangement"
UIDS = ("fork", "spoon", "plate")
RIGID = ("fork", "spoon", "plate")
ARTS = ()
ART_LINKS = {}
EEF = ()
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
PARALLEL_DEG = 15.0
FLAT_Z_M = 0.005
FLAT_DEG = 10.0
STILL_M = 0.0001
STILL_DEG = 0.1
ITEMS = ("fork", "spoon")
DEFINITION = ("on both sides = fork and spoon centres on opposite sides of the plate centre in world y and each farther than the plate radius from it; roughly parallel = long axes <= 15 deg apart in the table plane; "
              "placed stably = both touched by a robot at some point, no robot contact with either on this step, height back to the start flat height +- 5 mm, long axis < 10 deg from horizontal, "
              "per-step displacement <= 0.1 mm and turn <= 0.1 deg; hold = all of them for >= 20 consecutive steps up to the last step")


def geo_from_aabb(aabb):
    """aabb = {uid: [[lo xyz], [hi xyz]]} local bounding boxes with the episode's body scale applied (fork, spoon, plate) -> centres and plate radius."""
    cen = {u: 0.5 * (np.asarray(aabb[u][0], float) + np.asarray(aabb[u][1], float)) for u in ITEMS + ("plate",)}
    lo, hi = np.asarray(aabb["plate"][0], float), np.asarray(aabb["plate"][1], float)
    return dict(center=cen, plate_r=0.5 * float(np.sort(hi - lo)[-1]))     # plate diameter = longest edge of its box (the plate's vertical axis is the shortest)


class Judge:
    def __init__(self, geo):
        """geo = dict(center={fork, spoon, plate: local centre point}, plate_r=plate radius)."""
        self.g = geo
        self.z0, self.prev = None, None
        self.touched = {u: False for u in ITEMS}
        self.tail = C.Tail()

    def step(self, obs):
        """obs: T_fork, T_spoon, T_plate (4x4), touch_fork, touch_spoon (a robot link touches it on this step), t.  The first call is the start (both flat on the table)."""
        T = {u: np.asarray(obs["T_" + u], float) for u in ITEMS + ("plate",)}
        c = {u: C.world_point(T[u], self.g["center"][u]) for u in T}
        ax = {u: T[u][:3, 0] for u in ITEMS}
        if self.z0 is None:
            self.z0 = {u: float(c[u][2]) for u in ITEMS}
        rel = {u: (c[u] - c["plate"])[:2] for u in ITEMS}
        dist = {u: float(np.linalg.norm(rel[u])) for u in ITEMS}
        sides = bool(rel["fork"][1] * rel["spoon"][1] < 0 and all(dist[u] > self.g["plate_r"] for u in ITEMS))
        f2, s2 = ax["fork"][:2], ax["spoon"][:2]
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(f2 @ s2)) / (float(np.linalg.norm(f2)) * float(np.linalg.norm(s2)) + 1e-12)))))
        stable, mv, dz, tilt = True, {}, {}, {}
        for u in ITEMS:
            tch = bool(obs["touch_" + u])
            self.touched[u] |= tch
            dz[u] = float(c[u][2]) - self.z0[u]
            tilt[u] = float(np.degrees(np.arcsin(min(1.0, abs(float(C.unit(ax[u])[2]))))))
            if self.prev is None:
                mv[u] = float("nan")
                still = False
            else:
                mv[u] = float(np.linalg.norm(c[u] - self.prev[u][0]))
                still = mv[u] <= STILL_M and C.angle_between_deg(ax[u], self.prev[u][1]) <= STILL_DEG
            stable = stable and self.touched[u] and not tch and abs(dz[u]) <= FLAT_Z_M and tilt[u] < FLAT_DEG and still
        self.prev = {u: (c[u].copy(), ax[u].copy()) for u in ITEMS}
        conds = {"on_both_sides": sides, "roughly_parallel": bool(ang <= PARALLEL_DEG), "placed_stably": bool(stable)}
        done = C.tri(conds) is True
        n = self.tail.push(done, obs.get("t"))
        return dict(done=done, hold_n=n, conds=conds,
                    m=dict(angle_deg=ang, plate_radius_mm=self.g["plate_r"] * 1e3, fork_to_plate_mm=dist["fork"] * 1e3, spoon_to_plate_mm=dist["spoon"] * 1e3,
                           fork_rel_plate_y_mm=float(rel["fork"][1]) * 1e3, spoon_rel_plate_y_mm=float(rel["spoon"][1]) * 1e3,
                           fork_dz_mm=dz["fork"] * 1e3, spoon_dz_mm=dz["spoon"] * 1e3, fork_axis_tilt_deg=tilt["fork"], spoon_axis_tilt_deg=tilt["spoon"],
                           fork_step_move_mm=mv["fork"] * 1e3, spoon_step_move_mm=mv["spoon"] * 1e3, both_carried=all(self.touched.values())))


def make_judge(ctx):
    parts = P.parts_of(ctx.parts)
    return Judge(geo_from_aabb({u: parts[u]["box"]["aabb"] for u in ITEMS + ("plate",)}))


def obs_from_scene(scene, ctx=None):
    if not scene.has("obj_fork_x"):
        return None
    return dict(T_fork=scene.obj_T("fork"), T_spoon=scene.obj_T("spoon"), T_plate=scene.obj_T("plate"),
                touch_fork=bool(scene.robot_touching("fork")), touch_spoon=bool(scene.robot_touching("spoon")), t=scene.t)
