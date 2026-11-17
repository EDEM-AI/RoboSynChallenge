"""Unit tests for faq_checks.common and faq_checks.parts (numpy only; no simulator needed).

Run from the repository root:  python -m unittest discover -s tests/faq_checks -t .
"""
import unittest

import numpy as np

from faq_checks import TASKS, common as C, parts as P, task_module


class TestCommon(unittest.TestCase):
    def test_pose_roundtrip(self):
        T = C.T_from_pos_quat([1, 2, 3], [1, 0, 0, 0])
        self.assertTrue(np.allclose(T, np.eye(4) + np.array([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 0]])))
        # 90 degrees about z: x axis -> y axis
        q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        T = C.T_from_pos_quat([0, 0, 0], q)
        self.assertTrue(np.allclose(T[:3, 0], [0, 1, 0], atol=1e-12))

    def test_tilt_and_angle(self):
        self.assertAlmostEqual(C.tilt_deg([0, 0, 2]), 0.0)
        self.assertAlmostEqual(C.tilt_deg([1, 0, 0]), 90.0)
        self.assertAlmostEqual(C.tilt_deg([0, 0, -1]), 180.0)
        self.assertAlmostEqual(C.angle_between_deg([1, 0, 0], [0, 1, 0]), 90.0)
        with self.assertRaises(ValueError):
            C.unit([0, 0, 0])

    def test_tail_counts_and_resets_on_gap(self):
        t = C.Tail()
        self.assertEqual([t.push(True, i) for i in range(3)], [1, 2, 3])
        self.assertEqual(t.push(True, 10), 1)          # step number jumped -> restart
        self.assertEqual(t.push(False, 11), 0)

    def test_tri(self):
        self.assertIs(C.tri({"a": True, "b": True}), True)
        self.assertIs(C.tri({"a": True, "b": None}), None)
        self.assertIs(C.tri({"a": False, "b": None}), False)

    def test_summarize_needs_hold_and_last_step(self):
        ok = {"a": True}
        self.assertTrue(C.summarize("t", ok, C.HOLD_STEPS, 50, {}, "")["faq_success"])
        self.assertFalse(C.summarize("t", ok, C.HOLD_STEPS - 1, 50, {}, "")["faq_success"])
        self.assertFalse(C.summarize("t", {"a": None}, C.HOLD_STEPS, 50, {}, "")["faq_success"])


class TestParts(unittest.TestCase):
    def test_unit_scale_is_identity_for_points_and_lengths(self):
        part = {"kind": "disk", "p": [0.1, 0.2, 0.3], "normal": [0, 0, 1], "section": {"shape": "circle", "r_in": 0.01, "r_out": 0.02}}
        out = P.scale_parts({"mouth": part}, [1, 1, 1])["mouth"]
        self.assertEqual(out["p"], [0.1, 0.2, 0.3])
        self.assertEqual(out["normal"], [0.0, 0.0, 1.0])
        self.assertEqual(out["section"], {"shape": "circle", "r_in": 0.01, "r_out": 0.02})

    def test_scaling_rules(self):
        part = {"kind": "disk", "p": [1, 1, 1], "normal": [0, 0, 1],
                "section": {"shape": "rect", "in_or_out": "in", "w": 0.2, "d": 0.1, "axes": [[1, 0, 0], [0, 1, 0]]}}
        out = P.scale_parts({"x": part}, [2.0, 0.5, 4.0])["x"]
        self.assertEqual(out["p"], [2.0, 0.5, 4.0])                       # diag(s) p
        self.assertEqual(out["normal"], [0.0, 0.0, 1.0])                  # axis-aligned normal stays
        self.assertAlmostEqual(out["section"]["w"], 0.4)                  # w along x * 2
        self.assertAlmostEqual(out["section"]["d"], 0.05)                 # d along y * 0.5
        circ = P.scale_parts({"c": {"kind": "disk", "normal": [0, 0, 1], "section": {"shape": "circle", "r_out": 1.0}}}, [2, 3, 1])["c"]["section"]
        self.assertEqual(circ["shape"], "ellipse")                        # unequal cross axes -> ellipse
        self.assertEqual((circ["a_out"], circ["b_out"]), (2.0, 3.0))

    def test_unscaled_joint_range_flag(self):
        j = {"kind": "joint", "axis": [0, 0, 1], "range": [-0.005, 0.0], "unit": "m"}
        self.assertAlmostEqual(P.scale_parts({"press": j}, [1.1, 1.1, 1.1])["press"]["travel"], 0.0055)
        j["scaled"] = False
        self.assertAlmostEqual(P.scale_parts({"press": j}, [1.1, 1.1, 1.1])["press"]["travel"], 0.005)

    def test_table_covers_every_task_and_every_judge_builds(self):
        table = P.load_table()
        self.assertEqual(set(table["tasks"]), set(TASKS))
        for task in TASKS:
            m = task_module(task)
            ctx = P.EpisodeContext(task, {}, table={"top_z": 0.825, "xy_min": [0.2, -0.5], "xy_max": [1.2, 0.5]},
                                   limits={"pipette_slide_lo": -0.002})
            judge = m.make_judge(ctx)
            self.assertTrue(hasattr(judge, "step"), task)
            self.assertEqual(m.TASK, task)
            self.assertIsInstance(m.DEFINITION, str)

    def test_episode_parts_follow_body_scale(self):
        a = P.parts_for_episode("water_pouring", {"bottle": [1, 1, 1], "cup": [1, 1, 1]})
        b = P.parts_for_episode("water_pouring", {"bottle": [2, 2, 2], "cup": [1, 1, 1]})
        self.assertTrue(np.allclose(np.array(b["bottle"]["mouth"]["p"]), 2 * np.array(a["bottle"]["mouth"]["p"]), atol=3e-6))
        self.assertEqual(a["cup"], b["cup"])

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(KeyError):
            task_module("no_such_task")


if __name__ == "__main__":
    unittest.main()
