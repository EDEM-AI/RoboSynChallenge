"""Hand-built scenarios that drive each task module through ``obs_from_scene`` and ``Judge`` for the sentences that no recorded fixture covers
(a complete success for drawer_open_place, manipulate_pipette, water_pouring, and the switch sentences of mixer_operating).
Every scenario is run once as a success and once with one sentence broken."""
import unittest

import numpy as np

from faq_checks import common as C, parts as P, task_module
from faq_checks.scene import FlatScene

HOLD = C.HOLD_STEPS


def quat(R):
    """Rotation matrix -> (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0] * 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def put(state, prefix, pos, R=None):
    q = quat(np.eye(3) if R is None else R)
    for a, v in zip(("x", "y", "z", "qw", "qx", "qy", "qz"), [*pos, *q]):
        state["%s_%s" % (prefix, a)] = float(v)


def run(task, frames, body_scale, extra_parts=None, **ctx_kw):
    """frames: list of (state dict, contacts list). Returns (last Judge output, summary row)."""
    m = task_module(task)
    parts = P.parts_for_episode(task, body_scale)
    parts.update(extra_parts or {})
    ctx = P.EpisodeContext(task, body_scale, parts=parts, **ctx_kw)
    judge = m.make_judge(ctx)
    last = None
    for t, (state, contacts) in enumerate(frames):
        obs = m.obs_from_scene(FlatScene(state, contacts, (), t), ctx)
        if obs is not None:
            last = judge.step(obs)
    return last, C.summarize(task, last["conds"], last["hold_n"], len(frames), {}, "")


class TestDrawer(unittest.TestCase):
    def frames(self, opening=0.19, duck_xy=(0.0, 0.0), touch=False):
        out = []
        for t in range(30):
            s = {}
            put(s, "obj_duck", (*duck_xy, 0.0))
            put(s, "artlink_drawer_inner_box", (0.0, 0.0, 0.0))
            s["art_drawer_qpos_0"] = opening * min(1.0, t / 5.0)
            out.append((s, [["duck", "robot:right_link7", 3]] if touch else []))
        return out

    def test_success(self):
        _, row = run("drawer_open_place", self.frames(), {"duck": [0.8] * 3, "drawer": [1, 1, 0.35]})
        self.assertTrue(row["faq_success"], row["faq"])

    def test_each_sentence_can_fail(self):
        sc = {"duck": [0.8] * 3, "drawer": [1, 1, 0.35]}
        for kw, sentence in (({"opening": 0.1}, "drawer_fully_opened"), ({"duck_xy": (0.3, 0.0)}, "tomato_in_cavity"), ({"touch": True}, "hand_off_tomato")):
            _, row = run("drawer_open_place", self.frames(**kw), sc)
            self.assertFalse(row["faq_success"], sentence)
            self.assertFalse(row["faq"][sentence] is True, sentence)

    def test_drawer_pushed_back_is_not_open_at_the_end(self):
        frames = self.frames()
        for state, _ in frames[-10:]:
            state["art_drawer_qpos_0"] = 0.05
        _, row = run("drawer_open_place", frames, {"duck": [0.8] * 3, "drawer": [1, 1, 0.35]})
        self.assertIs(row["faq"]["still_open_at_end"], False)


class TestPipette(unittest.TestCase):
    def frames(self, press=True, release=True, touch_on_press=True):
        Rp = Ry(-np.pi / 2)                              # pipette local +x points up
        out = []

        def frame(root_z, q, touch):
            s = {}
            put(s, "art_pipette", (-0.003 if root_z > 0.25 else 0.5, 0.0, root_z), Rp)
            s["art_pipette_qpos_0"] = q
            put(s, "obj_beaker1", (0.0, 0.0, 0.0))
            put(s, "link_right_eef", (s["art_pipette_x"] + 0.05, 0.0, root_z - 0.1))
            rows = [["pipette:pipette_button_link", "robot:right_link8", 2]] if touch else []
            return s, rows

        out.append(frame(0.2, 0.01, False))                        # start, on the table
        out.append(frame(0.302, 0.01, False))                      # picked up, tip above the beaker mouth
        out += [frame(0.302, 0.01, False)] * 2                     # plateau readings before the press
        out.append(frame(0.302, -0.002 if press else 0.01, touch_on_press))
        out += [frame(0.302, 0.01 if release else -0.002, False)] * 25
        return out

    def go(self, **kw):
        return run("manipulate_pipette", self.frames(**kw), {"pipette": [3, 3, 3], "beaker1": [1.4, 1.4, 1.5]}, limits={"pipette_slide_lo": -0.002})

    def test_success(self):
        _, row = self.go()
        self.assertTrue(row["faq_success"], row["faq"])

    def test_no_press_no_release(self):
        _, row = self.go(press=False)
        self.assertFalse(row["faq_success"])
        self.assertIs(row["faq"]["button_pressed"], False)
        _, row = self.go(release=False)
        self.assertIs(row["faq"]["button_released"], False)

    def test_press_without_finger_contact_is_not_a_press(self):
        _, row = self.go(touch_on_press=False)
        self.assertIs(row["faq"]["button_pressed"], False)


class TestWater(unittest.TestCase):
    sc = {"bottle": [0.15] * 3, "cup": [0.75, 0.75, 1.0]}
    table = {"top_z": 0.825, "xy_min": [0.225, -0.5], "xy_max": [1.225, 0.5]}

    def frames(self, pour=True, touch_at_end=False):
        up = Rx(np.pi / 2)
        bottle0, cup0 = np.array([0.55, -0.1, 0.83]), np.array([0.7, 0.1, 0.85])
        cup_mouth = cup0 + np.array([0.003283, -0.0098, 0.043878])
        pour_pose = (cup_mouth + np.array([-0.02, 0.0, 0.05]) - np.array([0.23573, 0.0, 0.0]), Rz(-np.pi / 2))
        poses = [(bottle0, up, cup0), (bottle0, up, cup0 + [0, 0, 0.02])]
        poses.append((pour_pose[0], pour_pose[1], cup0) if pour else (bottle0, up, cup0))
        poses += [(bottle0, up, cup0)] * 30
        out = []
        for i, (b, R, c) in enumerate(poses):
            s = {}
            put(s, "obj_bottle", b, R)
            put(s, "obj_cup", c)
            put(s, "link_right_eef", b + [0.0, 0.0, 0.1])
            rows = [["bottle", "robot:right_link7", 2]] if (touch_at_end and i == len(poses) - 1) else []
            out.append((s, rows))
        return out

    def go(self, **kw):
        return run("water_pouring", self.frames(**kw), self.sc, table=self.table)

    def test_success(self):
        _, row = self.go()
        self.assertTrue(row["faq_success"], row["faq"])

    def test_no_pour(self):
        _, row = self.go(pour=False)
        self.assertIs(row["faq"]["poured"], False)
        self.assertFalse(row["faq_success"])

    def test_hand_still_on_the_bottle(self):
        _, row = self.go(touch_at_end=True)
        self.assertFalse(row["faq_success"])


class TestMixerSwitch(unittest.TestCase):
    sc = {"beaker": [1.4, 1.4, 1.5], "beaker_mixer": [0.3] * 3}
    switch = {"mixer_switch": {"flip": {"kind": "joint", "range": [0.0, 0.523599], "unit": "rad"}}}

    def frames(self, with_switch=True, press=True):
        beaker_on_ring = (0.00373, 0.0357, -0.028)
        plan = [((0.5, 0.3, 0.0), [["beaker", "table", 1]], 0.0),
                ((0.5, 0.3, 0.1), [["beaker", "robot:right_link7", 1]], 0.0),
                (beaker_on_ring, [["beaker", "beaker_mixer", 1], ["beaker", "robot:right_link7", 1]], 0.0),
                (beaker_on_ring, [["beaker", "beaker_mixer", 1]], 0.0),
                (beaker_on_ring, [["beaker", "beaker_mixer", 1], ["mixer_switch:switch_lever", "robot:left_link7", 1]], 0.1),
                (beaker_on_ring, [["beaker", "beaker_mixer", 1], ["mixer_switch:switch_lever", "robot:left_link7", 1]], 0.5 if press else 0.1)]
        plan += [(beaker_on_ring, [["beaker", "beaker_mixer", 1]], 0.5 if press else 0.1)] * 25
        out = []
        for pos, rows, q in plan:
            s = {}
            put(s, "obj_beaker", pos)
            put(s, "obj_beaker_mixer", (0.0, 0.0, 0.0))
            if with_switch:
                s["art_mixer_switch_qpos_0"] = q
            out.append((s, rows))
        return out

    def test_success_with_switch(self):
        _, row = run("mixer_operating", self.frames(), self.sc, extra_parts=self.switch)
        self.assertTrue(row["faq_success"], row["faq"])

    def test_switch_not_pressed(self):
        _, row = run("mixer_operating", self.frames(press=False), self.sc, extra_parts=self.switch)
        self.assertFalse(row["faq_success"])
        self.assertIs(row["faq"]["button_pressed"], False)

    def test_without_a_switch_the_switch_sentences_are_unknown(self):
        _, row = run("mixer_operating", self.frames(with_switch=False), self.sc)
        self.assertFalse(row["faq_success"])
        for k in ("button_pressed", "switch_on_hand_off", "order"):
            self.assertIsNone(row["faq"][k], k)
        self.assertTrue(row["faq"]["beaker_lifted"] and row["faq"]["on_circle"] and row["faq"]["resting_stably"])


if __name__ == "__main__":
    unittest.main()
