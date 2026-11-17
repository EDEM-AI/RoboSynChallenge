"""Offline tests for drawer_handle.py (no official assets needed)."""
import os
import tempfile
import unittest

import numpy as np

import drawer_handle as D


class TestDrawerHandle(unittest.TestCase):
    def test_block_counts_and_convexity(self):
        for fam, n in (("shallow_arc", 18), ("bridge", 3)):
            blocks = D.FAMILIES[fam]()
            self.assertEqual(len(blocks), n)
            for b in blocks:
                self.assertTrue(b.is_watertight)
                self.assertAlmostEqual(b.volume, b.convex_hull.volume, places=12)   # each block is convex
                self.assertGreaterEqual(b.bounds[0][0], D.FACE_X - 1e-12)          # nothing behind the front panel

    def test_width_is_80mm(self):
        for fam, fn in D.FAMILIES.items():
            b = np.vstack([m.vertices for m in fn()])
            self.assertAlmostEqual(float(np.ptp(b[:, 1])), 0.08, places=9, msg=fam)

    def test_obj_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            blocks = D.bridge_blocks()
            p = os.path.join(d, "h.obj")
            D.write_obj(p, blocks, prefix="handle")
            back = D.read_groups(p)
            self.assertEqual(len(back), len(blocks))
            for (_, m), b in zip(back, blocks):
                self.assertTrue(np.allclose(m.vertices, b.vertices, atol=1e-9))

    def test_deterministic(self):
        a = np.vstack([m.vertices for m in D.shallow_arc_blocks()])
        b = np.vstack([m.vertices for m in D.shallow_arc_blocks()])
        self.assertTrue((a == b).all())


if __name__ == "__main__":
    unittest.main()
