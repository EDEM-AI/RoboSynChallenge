"""Tests for the library-import shapes (: bell / plate / fork / spoon).

Separate from test_shape_gen.py so the two files do not edit the
same file.  Run:

    python3 -m unittest test_ware -v
"""

import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

import geom
import ware_ready as WR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "libimport")
SEL = os.path.join(HERE, "selection.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def walk(o, path="$"):
    """Yield (path, dict) for every parts entry."""
    if isinstance(o, dict):
        if "kind" in o and ("src" in o or "tol" in o):
            yield path, o
        for k, val in o.items():
            yield from walk(val, path + "." + str(k))
    elif isinstance(o, list):
        for i, val in enumerate(o):
            yield from walk(val, f"{path}[{i}]")


class TestWareReady(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SEL, encoding="utf-8") as fh:
            cls.sel = json.load(fh)
        cls.built = {}

    def build(self, row):
        key = (row["cls"], row["key"], row["target_m"])
        if key not in self.built:
            self.built[key] = WR.build(row["cls"], row["key"], row["target_m"], row["name"])
        return self.built[key]

    def test_relations_all_pass(self):
        for row in self.sel:
            with self.subTest(v=row["name"]):
                _v, _f, ann, _m = self.build(row)
                bad = [r["id"] for r in ann["relations"] if not r["ok"]]
                self.assertEqual(bad, [], f"{row['name']} failed {bad}")

    def test_parts_schema(self):
        for row in self.sel:
            with self.subTest(v=row["name"]):
                _v, _f, ann, _m = self.build(row)
                self.assertTrue(ann["parts_schema"].startswith("hq-ctx-parts/3"))
                seen = 0
                for path, part in walk(ann["parts"]):
                    seen += 1
                    self.assertIn(part["kind"], ("point", "axis", "disk", "region",
                                                 "joint", "list"), path)
                    self.assertIn("src", path and part)
                    if part["kind"] == "joint":
                        self.assertTrue(0 < part["travel"])
                        self.assertEqual(part["travel"],
                                         part["range"][1] - part["range"][0])
                        self.assertIn(part["unit"], ("m", "rad"))
                    if "axis" in part:
                        self.assertAlmostEqual(float(np.linalg.norm(part["axis"])), 1.0, 6)
                    if "normal" in part:
                        self.assertAlmostEqual(float(np.linalg.norm(part["normal"])), 1.0, 6)
                    if part["kind"] == "list":
                        ids = [it["id"] for it in part["items"]]
                        self.assertEqual(ids, list(range(len(ids))), path)
                        for it in part["items"]:
                            self.assertIn("src", it)
                            self.assertIn("tol", it)
                self.assertGreater(seen, 3)

    def test_bbox_and_com(self):
        for row in self.sel:
            with self.subTest(v=row["name"]):
                v, _f, ann, _m = self.build(row)
                lo, hi = ann["bbox"]["min"], ann["bbox"]["max"]
                self.assertTrue(np.allclose(lo, v.min(0), atol=1e-5))
                self.assertTrue(np.allclose(hi, v.max(0), atol=1e-5))
                ext = np.asarray(ann["bbox"]["extent"])
                self.assertTrue(np.allclose(ext, v.max(0) - v.min(0), atol=1e-5))
                units = list(ann["parts"].values())[0]
                if "com" in units:
                    self.assertTrue(np.allclose(units["com"]["p"], ann["mass"]["com"],
                                                atol=2e-6))

    def test_mass_sanity(self):
        """回归：质量 > 0、惯量对角 > 0 且满足三角不等式、质心在包围盒内。"""
        for row in self.sel:
            with self.subTest(v=row["name"]):
                v, _f, ann, _m = self.build(row)
                mass = ann["mass"]["mass_kg"]
                self.assertGreater(mass, 0.0)
                d = np.diag(np.asarray(ann["mass"]["inertia_kgm2"], dtype=float))
                self.assertTrue((d > 0).all(), d)
                self.assertGreaterEqual(d[0] + d[1], d[2] - 1e-12)
                self.assertGreaterEqual(d[0] + d[2], d[1] - 1e-12)
                self.assertGreaterEqual(d[1] + d[2], d[0] - 1e-12)
                lo, hi = ann["bbox"]["min"], ann["bbox"]["max"]
                com = ann["mass"]["com"]
                for i in range(3):
                    self.assertGreaterEqual(com[i], lo[i] - 1e-9)
                    self.assertLessEqual(com[i], hi[i] + 1e-9)
                self.assertIn("estimate", ann["mass"]["source"].lower()
                              if "estimate" in ann["mass"]["source"].lower()
                              else ann["mass"]["source"])

    def test_frame(self):
        for row in self.sel:
            with self.subTest(v=row["name"]):
                v, _f, ann, _m = self.build(row)
                cls = row["cls"]
                if cls in ("bell", "plate"):
                    self.assertAlmostEqual(float(v[:, 2].min()), 0.0, 9)
                    for ax in (0, 1):
                        self.assertAlmostEqual(float(v[:, ax].min() + v[:, ax].max()), 0.0, 9)
                else:
                    self.assertAlmostEqual(float(v[:, 0].min() + v[:, 0].max()), 0.0, 9)
                    head = ann["parts"][cls]["head"]["p"][0]
                    self.assertGreater(head, 0.0)

    def test_mesh_parts_cover_faces(self):
        for row in self.sel:
            with self.subTest(v=row["name"]):
                _v, f, ann, _m = self.build(row)
                runs = ann["mesh_parts"]
                self.assertEqual(runs[0]["face_start"], 0)
                self.assertEqual(runs[-1]["face_end"], len(f))
                for a, b in zip(runs, runs[1:]):
                    self.assertEqual(a["face_end"], b["face_start"])

    def test_mesh_parts_granularity(self):
        for row in self.sel:
            with self.subTest(v=row["name"]):
                _v, _f, ann, _m = self.build(row)
                runs = ann["mesh_parts"]
                if len(runs) > 64:
                    self.fail(f"{row['name']} has {len(runs)} mesh_parts")
                if len(runs) == 1 and runs[0]["name"] == "all":
                    self.assertTrue(ann.get("mesh_parts_note"), row["name"])

    def test_no_field_table_extension(self):
        for row in self.sel:
            if row["cls"] not in ("fork", "spoon"):
                continue
            with self.subTest(v=row["name"]):
                _v, _f, ann, _m = self.build(row)
                keys = set(ann["parts"][row["cls"]])
                self.assertEqual(keys, {"origin", "head", "tail", "body", "handle", "com"})

    def test_deterministic(self):
        for row in self.sel:
            with self.subTest(v=row["name"]):
                with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
                    WR.write_variant(row["cls"], row["index"], row["name"], row["key"],
                                     row["target_m"], d1)
                    WR.write_variant(row["cls"], row["index"], row["name"], row["key"],
                                     row["target_m"], d2)
                    a1 = os.path.join(d1, row["name"])
                    a2 = os.path.join(d2, row["name"])
                    files = sorted(os.listdir(a1))
                    self.assertEqual(files, sorted(os.listdir(a2)))
                    for fn in files:
                        self.assertEqual(sha(os.path.join(a1, fn)), sha(os.path.join(a2, fn)), fn)

    @unittest.skipUnless(os.path.isdir(os.path.join(HERE, "out", "libimport")), "needs generated ./out/libimport")
    def test_index_and_licence(self):
        allowed = {"by", "cc0", "CC-BY-4.0", "Apache-2.0"}
        for cls in ("bell", "plate", "fork", "spoon"):
            p = os.path.join(OUT, cls, "index.json")
            self.assertTrue(os.path.exists(p), p)
            with open(p, encoding="utf-8") as fh:
                idx = json.load(fh)
            self.assertGreaterEqual(len(idx["rows"]), 5, cls)
            for r in idx["rows"]:
                self.assertIn(r["license"], allowed)
                for fn, want in r["files"].items():
                    got = sha(os.path.join(OUT, cls, r["variant"], fn))
                    self.assertEqual(got, want, f"{cls}/{r['variant']}/{fn}")

    @unittest.skipUnless(os.path.isdir(os.path.join(HERE, "out", "libimport")), "needs generated ./out/libimport")
    def test_source_keys(self):
        for cls in ("bell", "plate", "fork", "spoon"):
            path = os.path.join(OUT, cls, "index.json")
            with open(path, encoding="utf-8") as fh:
                idx = json.load(fh)
            for r in idx["rows"]:
                with open(os.path.join(OUT, cls, r["variant"],
                                       r["variant"] + ".variant.yaml"), encoding="utf-8") as fh:
                    yml = fh.read()
                for key in ("base_sha256", "license", "source_kind", "import_sha",
                            "functional_read", "status: draft"):
                    self.assertIn(key, yml, f"{r['variant']} missing {key}")


class TestNoSimulatorImport(unittest.TestCase):
    def test_import_lines(self):
        banned = ("torch", "embodichain", "sapien", "robosynchallenge", "dexsim")
        for fn in ("ware_ready.py", "ware_select.py"):
            with open(os.path.join(HERE, fn), encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s.startswith("import ") or s.startswith("from "):
                        self.assertFalse(any(b in s for b in banned), f"{fn}: {s}")

    def test_import_step_is_declared(self):
        txt = open(os.path.join(HERE, "ware_import.py"), encoding="utf-8").read()
        self.assertIn("the ONLY place that touches trimesh", txt.split('"""')[1])


if __name__ == "__main__":
    unittest.main()
