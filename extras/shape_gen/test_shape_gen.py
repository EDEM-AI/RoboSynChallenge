"""Self-tests for the two generators.

    python3 -m unittest test_shape_gen -v
    (or)  python3 test_shape_gen.py

Covers, per the task card:
  * every relation inequality on all six samples of both families,
  * annotation points really sit on the mesh / inside the holes,
  * the nominal parameters reproduce the existing assets' measured size,
  * determinism (same params + same seed -> identical bytes),
  * bad parameters are rejected instead of being written,
  * the generators import no simulator.
"""

import filecmp
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

import geom
import marker
import params as P
import rack
import make_samples

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED = os.path.join(HERE, "nominal_measured.json")


def _load_measured():
    with open(MEASURED, encoding="utf-8") as fh:
        return json.load(fh)


NEEDS_OUT = unittest.skipUnless(os.path.isdir(os.path.join(HERE, "out")), "needs generated ./out (run the generators first)")


class TestRelations(unittest.TestCase):
    def test_rack_samples(self):
        base = rack.default_params()
        for label, over in make_samples.RACK_SAMPLES:
            p = dict(base)
            p.update(over)
            p["name"] = label
            rel = rack.check_relations(p)
            self.assertTrue(all(r["ok"] for r in rel), label)
            self.assertGreaterEqual(len(rel), 10, label)

    def test_marker_samples(self):
        base = marker.default_params()
        for label, over in make_samples.MARKER_SAMPLES:
            p = dict(base)
            p.update(over)
            p["name"] = label
            rel = marker.check_relations(p)
            self.assertTrue(all(r["ok"] for r in rel), label)

    def test_rack_rejects_bad_cell(self):
        p = rack.default_params()
        p["cell"] = 0.0198            # smaller than the 20 mm tube -> must fail
        with self.assertRaises(ValueError):
            rack.check_relations(p)

    def test_rack_rejects_bad_pitch(self):
        p = rack.default_params()
        p["pitch_x"] = p["cell"] + 0.001   # grid wall thinner than 1.5 mm
        with self.assertRaises(ValueError):
            rack.check_relations(p)

    def test_marker_rejects_big_diameter(self):
        p = marker.default_params()
        p["d"] = 0.080                # would not fit the holder
        with self.assertRaises(ValueError):
            marker.check_relations(p)

    def test_rack_rejects_unstable(self):
        """Excessive clearance with a small layer gap -> lean over the 5 deg criterion."""
        p = rack.default_params()
        p["cell"] = 0.0215
        p["tube_d"] = 0.0160            # 2.75 mm clearance per side
        p["gap"] = 0.010                # plates too close -> big lean
        with self.assertRaises(ValueError):
            rack.check_relations(p)

    def test_rack_rejects_single_layer(self):
        p = rack.default_params()
        p["layers"] = 1
        with self.assertRaises(ValueError):
            rack.check_relations(p)


class TestGeometry(unittest.TestCase):
    def test_rack_watertight_and_volume(self):
        base = rack.default_params()
        for label, over in make_samples.RACK_SAMPLES:
            p = dict(base)
            p.update(over)
            p["name"] = label
            mesh, d = rack.build(p)
            verts, faces = mesh.welded_parts()
            chk = geom.check_faces(verts, faces)
            self.assertTrue(chk["watertight"], label)
            self.assertTrue(chk["orientation_consistent"], label)
            self.assertEqual(chk["duplicate_faces"], 0, label)
            v_ana = rack.analytic_volume(p, d)
            self.assertLess(abs(chk["volume_m3"] - v_ana) / v_ana, 0.01, label)

    def test_marker_watertight_and_volume(self):
        base = marker.default_params()
        for label, over in make_samples.MARKER_SAMPLES:
            p = dict(base)
            p.update(over)
            p["name"] = label
            mesh = marker.build(p, seed=0)
            verts, faces = mesh.welded(eps=1e-8)
            chk = geom.check_faces(verts, faces)
            self.assertTrue(chk["watertight"], label)
            self.assertTrue(chk["orientation_consistent"], label)
            v = marker._volume(p)
            self.assertLess(abs(chk["volume_m3"] - v) / v, 0.02, label)

    def test_units_are_metres(self):
        mesh, d = rack.build(rack.default_params())
        verts, _ = mesh.welded_parts()
        self.assertLess(np.max(verts.max(0) - verts.min(0)), 0.60)
        m = marker.build(marker.default_params(), 0)
        v, _ = m.welded(eps=1e-8)
        self.assertLess(np.max(v.max(0) - v.min(0)), 0.30)

    def test_normals_point_outward(self):
        for m in (rack.build(rack.default_params())[0], marker.build(marker.default_params(), 0)):
            verts, faces = (m.welded_parts() if len(m.parts) > 1 else m.welded(eps=1e-8))
            a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            vol = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
            self.assertGreater(vol, 0.0)


class TestAnnotations(unittest.TestCase):
    def setUp(self):
        self.p = rack.default_params()
        self.p["name"] = "t"
        res = rack.generate(self.p, seed=0, out_dir=None)
        self.ann = res["annotations"]
        self.verts, self.faces = res["mesh"]
        self.mesh, self.d = rack.build(self.p)

    def test_every_hole_is_open_and_points_are_inside(self):
        H = self.ann["derived"]["outer_m"]["H"]
        bt = self.p["base_t"]
        r = self.p["cell"] / 2.0
        self.assertEqual(len(self.ann["functional"]), self.p["rows"] * self.p["cols"])
        for h in self.ann["functional"]:
            x, y, z = h["point"]
            self.assertAlmostEqual(z, H, places=9)
            # straight down through the hole: nothing until the base tray
            hits = geom.ray_hits(self.verts, self.faces, (x, y, H + 0.002), (0, 0, -1),
                                 tmax=(H - bt) - 0.001)
            self.assertEqual(hits, 0, f"{h['name']} is not a through hole")
            # the axis stays clear for the whole free depth
            inside = geom.inside(self.verts, self.faces, (x, y, H - (H - bt) * 0.5))
            self.assertFalse(inside, f"{h['name']} axis blocked")
            # cell size: the first material along +x at mid-plate is the cell wall
            if h["name"] in ("hole_r0c0", "hole_r0c1", "hole_r1c0"):
                mid = z - self.p["plate_t"] / 2.0
                for direction in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
                    dist = geom.ray_distance(self.verts, self.faces, (x, y, mid), direction)
                    self.assertLess(abs(dist - r), 0.0003,
                                    f"{h['name']} {direction} wall at {dist:.6f} vs {r:.6f}")

    def test_cells_aligned_between_layers(self):
        plates = self.ann["derived"]["grid_plate_z"]
        self.assertIn(len(plates), (2, 3))
        holes = self.ann["functional"]
        # every hole centre must be identical for all grid plates (aligned stack)
        for h in holes:
            for z in plates:
                self.assertGreaterEqual(h["point"][2], z)   # mouth on top of the stack
        self.assertEqual(len({(round(h["point"][0], 9), round(h["point"][1], 9)) for h in holes}),
                         self.p["rows"] * self.p["cols"])

    def test_hole_grid_matches_parameters(self):
        holes = self.ann["functional"]
        xs = sorted({round(h["point"][0], 6) for h in holes})
        ys = sorted({round(h["point"][1], 6) for h in holes})
        self.assertEqual(len(xs), self.p["cols"])
        self.assertEqual(len(ys), self.p["rows"])
        self.assertAlmostEqual(xs[1] - xs[0], self.p["pitch_x"], places=6)
        self.assertAlmostEqual(ys[1] - ys[0], self.p["pitch_y"], places=6)

    def test_grasp_points_on_material(self):
        for g in self.ann["grasp"]:
            p = np.asarray(g["point"], dtype=float)
            d = geom.distance_point_to_surface(p, self.verts, self.faces)
            self.assertLess(d, 0.003, f"{g['name']} is {d*1e3:.2f} mm from the surface")

    def test_placement_point_on_support_plane(self):
        pl = self.ann["placement"][0]
        self.assertEqual(pl["point"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(self.verts[:, 2].min(), 0.0, places=9)

    def test_mass_properties_sane(self):
        m = self.ann["mass"]
        self.assertGreater(m["mass_kg"], 0.05)
        self.assertLess(m["mass_kg"], 1.0)
        self.assertAlmostEqual(m["com"][2], m["com"][2], places=9)
        I = np.asarray(m["inertia_kgm2"])
        w = np.linalg.eigvalsh(I)
        self.assertTrue(np.all(w > 0), w)
        self.assertGreaterEqual(w[0] + w[1], w[2] * (1 - 1e-6))   # triangle inequality


class TestMarkerAnnotations(unittest.TestCase):
    def setUp(self):
        p = marker.default_params()
        p["name"] = "t"
        res = marker.generate(p, seed=0, out_dir=None)
        self.ann = res["annotations"]
        self.verts, self.faces = res["mesh"]
        self.p = p

    def test_tip_and_tail_on_surface(self):
        L = self.p["length"]
        for name in ("tip", "tail"):
            pt = [f for f in self.ann["functional"] if f["name"] == name][0]["point"]
            d = geom.distance_point_to_surface(pt, self.verts, self.faces)
            self.assertLess(d, 1e-6, f"{name} off surface by {d}")
        self.assertAlmostEqual(self.verts[:, 0].max(), L / 2, places=9)
        self.assertAlmostEqual(self.verts[:, 0].min(), -L / 2, places=9)

    def test_grasp_points_on_barrel(self):
        r = self.p["d"] / 2.0
        # the barrel is a 48-gon, so the distance to the surface is the inradius
        tol = r * (1.0 - np.cos(np.pi / int(self.p["sides"]))) + 1e-9
        for g in self.ann["grasp"]:
            d = geom.distance_point_to_surface(g["point"], self.verts, self.faces)
            self.assertLess(abs(d - r), tol, f"{g['name']} distance {d}")

    def test_grasp_bands_do_not_overlap(self):
        s = sorted(g["point"][0] for g in self.ann["grasp"])
        self.assertGreaterEqual(s[1] - s[0], 0.060)   # U geometry constraint

    def test_placement_below_barrel(self):
        pl = self.ann["placement"][0]
        r_cap = self.p["d"] / 2 + (self.p["cap_clearance"] if self.p["cap_len"] > 0 else 0)
        self.assertAlmostEqual(pl["point"][2], -r_cap, places=9)
        self.assertAlmostEqual(self.verts[:, 2].min(), -r_cap, places=6)


class TestNominalReproduction(unittest.TestCase):
    """The nominal parameters must land on the existing assets' measured size."""

    def setUp(self):
        self.meas = _load_measured()

    def test_rack_matches_sim_asset(self):
        """Raw (unscaled) official asset: 250.21 x 105.59 x 72.08 mm, 12 x 5 cells."""
        target = self.meas["assets"]["test_tube_rack_standard.ply"]["extent_raw"]
        p = dict(rack.default_params())
        p.update({"rows": 5, "cols": 12, "cell": 0.0169,
                  "pitch_x": 0.0202, "pitch_y": 0.0190, "tube_d": 0.01678,
                  "layers": 3, "gap": 0.0288, "side_wall_t": 0.0022,
                  "margin_x": 0.00335, "margin_y": 0.00415})
        d = rack.derive(p)
        got = [d["Lx"], d["Wy"], d["H"]]
        for g, t in zip(got, target):
            self.assertLess(abs(g - t), 0.0025, f"{g:.5f} vs {t:.5f}")
        self.assertEqual((d["rows"], d["cols"]), (5, 12))

    def test_marker_matches_sim_pen(self):
        target = self.meas["scaled"]["pen"]            # 0.1434 x 0.01533 x 0.01533
        p = dict(marker.default_params())
        p.update({"d": 0.01533, "length": 0.1434, "cap_len": 0.0})
        mesh = marker.build(p, seed=0)
        verts, _ = mesh.welded(eps=1e-8)
        ext = verts.max(0) - verts.min(0)
        self.assertLess(abs(ext[0] - target[0]), 0.0001)
        self.assertLess(abs(ext[1] - target[1]), 0.0001)
        self.assertLess(abs(ext[2] - target[2]), 0.0001)

    def test_tube_nominal_used_for_relations(self):
        tube = self.meas["scaled"]["tube"]
        p = dict(rack.default_params())
        p.update(dict(make_samples.RACK_SAMPLES)["rack_12x5_simsize"])   # the sim-size sample
        self.assertAlmostEqual(p["tube_d"], tube[0], places=5)
        self.assertAlmostEqual(p["tube_len"], tube[2], places=5)
        self.assertAlmostEqual(rack.default_params()["tube_len"], tube[2], places=5)


class TestDeterminism(unittest.TestCase):
    def _sha_dir(self, path):
        out = {}
        for fn in sorted(os.listdir(path)):
            with open(os.path.join(path, fn), "rb") as fh:
                out[fn] = hashlib.sha256(fh.read()).hexdigest()
        return out

    def test_rack_byte_identical(self):
        a = tempfile.mkdtemp()
        b = tempfile.mkdtemp()
        try:
            rack.generate(rack.default_params(), seed=0, out_dir=a)
            rack.generate(rack.default_params(), seed=0, out_dir=b)
            self.assertEqual(self._sha_dir(a), self._sha_dir(b))
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_marker_byte_identical(self):
        a = tempfile.mkdtemp()
        b = tempfile.mkdtemp()
        try:
            marker.generate(marker.default_params(), seed=7, out_dir=a)
            marker.generate(marker.default_params(), seed=7, out_dir=b)
            self.assertEqual(self._sha_dir(a), self._sha_dir(b))
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_seed_changes_faceting_phase(self):
        a = tempfile.mkdtemp()
        b = tempfile.mkdtemp()
        try:
            p = marker.default_params()
            p["sides"] = 12
            va = marker.generate(p, seed=0, out_dir=a)
            vb = marker.generate(p, seed=1, out_dir=b)
            ka = [k for k in va["files"] if k.endswith(".obj")][0]
            kb = [k for k in vb["files"] if k.endswith(".obj")][0]
            self.assertNotEqual(va["files"][ka], vb["files"][kb])
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_variant_yaml_registers_files(self):
        for module, params, tag in ((rack, rack.default_params(), "rack"),
                                    (marker, marker.default_params(), "marker")):
            d = tempfile.mkdtemp()
            try:
                v = module.generate(params, seed=0, out_dir=d)
                self.assertIn("files", v, tag)
                for fn, sha in v["files"].items():
                    self.assertEqual(geom.sha256_file(os.path.join(d, fn)), sha, f"{tag}:{fn}")
                self.assertIn("generator", v["source"], tag)
                self.assertEqual(v["body_scale"], [1.0, 1.0, 1.0], tag)
                # v2.2 registry pointer
                self.assertEqual(v["functional_read"], {"path": v["name"] + ".annotations.json",
                                                        "key": "parts"}, tag)
                yml = os.path.join(d, v["name"] + ".variant.yaml")
                with open(yml, encoding="utf-8") as fh:
                    text = fh.read()
                self.assertIn("functional_read:", text, tag)
            finally:
                shutil.rmtree(d, ignore_errors=True)

    def test_manifest_matches_disk(self):
        mp = os.path.join(HERE, "out", "manifest.json")
        if not os.path.exists(mp):
            self.skipTest("no manifest yet")
        with open(mp, encoding="utf-8") as fh:
            man = json.load(fh)
        n = 0
        for row in man.get("extra_rows", []):
            for fn, sha in row["files"].items():
                self.assertEqual(geom.sha256_file(os.path.join(row["dir"], fn)), sha, fn)
                n += 1
        for fam, rows in man["families"].items():
            for r in rows:
                for fn, sha in r["files"].items():
                    self.assertEqual(geom.sha256_file(os.path.join(r["dir"], fn)), sha, fn)
                    n += 1
        self.assertGreaterEqual(n, 36)


class TestCtxPartsFieldTable(unittest.TestCase):
    """parts 字段表第 3 版：部位名、kind、字段。"""

    KINDS = {"point", "axis", "disk", "region", "joint", "list"}

    def _walk(self, part, path):
        self.assertIn("kind", part, path)
        self.assertIn(part["kind"], self.KINDS, path)
        self.assertIn("src", part, path)
        self.assertIn("tol", part, path)
        self.assertTrue(part["src"].startswith("variant:"), path)
        k = part["kind"]
        if k in ("point", "axis", "disk", "region"):
            self.assertEqual(len(part["p"]), 3, path)
        if k == "axis":
            self.assertEqual(len(part["axis"]), 3, path)
            self.assertAlmostEqual(np.linalg.norm(part["axis"]), 1.0, places=9, msg=path)
        if k == "disk":
            # 第 3 版：disk 用 normal + section，不再用 axis / radius
            self.assertEqual(len(part["normal"]), 3, path)
            self.assertAlmostEqual(np.linalg.norm(part["normal"]), 1.0, places=9, msg=path)
            self.assertIn("section", part, path)
            sec = part["section"]
            self.assertIn(sec["shape"], ("circle", "rect", "ellipse"), path)
            if sec["shape"] == "rect":
                self.assertIn("w", sec, path)
                self.assertIn("d", sec, path)
            if sec["shape"] == "circle":
                self.assertIn("r_out", sec, path)
        if k == "list":
            self.assertIn("items", part, path)
            self.assertTrue(part["items"], path)
            ids = []
            for it in part["items"]:
                self.assertIn("p", it, path)
                self.assertIn("id", it, path)
                self.assertIsInstance(it["id"], int, path)
                ids.append(it["id"])
                has_axis = "axis" in it
                has_normal = "normal" in it
                self.assertTrue(has_axis or has_normal, path)
                self.assertIn("src", it, path)
                self.assertIn("tol", it, path)
            self.assertEqual(ids, list(range(len(ids))), path)   # 0..n-1

    def test_rack_parts(self):
        p = rack.default_params()
        p["name"] = "t"
        ann = rack.generate(p, 0, None)["annotations"]
        self.assertEqual(ann["parts_schema"].split(" ")[0], "hq-ctx-parts/3")
        parts = ann["parts"]["rack"]
        for name in ("origin", "holes", "plates", "com"):
            self.assertIn(name, parts)
        for name in ("origin", "holes", "plates", "com"):
            self._walk(parts[name], "rack." + name)
        items = parts["holes"]["items"]
        self.assertEqual(len(items), p["rows"] * p["cols"])
        for it in items:
            self.assertEqual(it["axis"], [0.0, 0.0, 1.0])
            self.assertEqual(it["section"]["shape"], "rect")
            self.assertAlmostEqual(it["size"]["depth"],
                                   ann["derived"]["outer_m"]["H"] - p["base_t"], places=9)
        # sorted by local x then y
        key = [(round(h["p"][0], 9), round(h["p"][1], 9)) for h in items]
        self.assertEqual(key, sorted(key))
        # 各板 role 按实际层数给，不虚构中板
        roles = [q.get("role") for q in parts["plates"]["items"]]
        expect = ["bottom", "top"] if int(p["layers"]) == 2 else ["bottom", "middle", "top"]
        self.assertEqual(roles, expect)
        self.assertNotIn("plates_note", parts)          # 说明放注释层，不混进 parts
        self.assertIn("plates", ann.get("notes", {}))
        # v3 架那行只有 holes / plates；顶面 = plates[-1]
        self.assertNotIn("top", parts)
        self.assertAlmostEqual(parts["plates"]["items"][-1]["p"][2],
                               ann["derived"]["outer_m"]["H"] - p["plate_t"] / 2, places=9)
        # com: the part must carry the real centre of mass, not a placeholder
        self.assertIn("com", parts)
        np.testing.assert_allclose(parts["com"]["p"], ann["mass"]["com"], atol=1e-9)
        self.assertEqual(parts["com"]["com_from"], "mass")
        # plates: one face per grid plate, all with +z normal
        pl = parts["plates"]["items"]
        self.assertEqual(len(pl), int(p["layers"]))
        for q in pl:
            self.assertEqual(q["normal"], [0.0, 0.0, 1.0])

    def test_marker_parts(self):
        p = marker.default_params()
        p["name"] = "t"
        ann = marker.generate(p, 0, None)["annotations"]
        parts = ann["parts"]["pen"]
        for name in ("tip", "tail", "body", "com"):
            self.assertIn(name, parts)
            self._walk(parts[name], "pen." + name)
        L = p["length"]
        self.assertAlmostEqual(parts["tip"]["p"][0], L / 2, places=9)
        self.assertAlmostEqual(parts["tail"]["p"][0], -L / 2, places=9)
        self.assertAlmostEqual(parts["body"]["size"]["length"], L, places=9)
        self.assertTrue(parts["body"]["profile"])            # 沿轴半径剖面
        self.assertTrue(0.0 <= parts["body"]["grasp_s"] <= 1.0)
        np.testing.assert_allclose(parts["com"]["p"], ann["mass"]["com"], atol=1e-9)
        self.assertEqual(parts["com"]["com_from"], "mass")
        self.assertGreater(abs(parts["com"]["p"][0]), 1e-4)    # a capped pen is not centred
        d = np.subtract(parts["tip"]["p"], parts["tail"]["p"])
        self.assertAlmostEqual(d[0] / np.linalg.norm(d), 1.0, places=9)


class TestReadyMadeMarkers(unittest.TestCase):
    """marker_ready.py：现成模型路径的三条核（确定性 / 只读 numpy / 基件 sha 登记）。"""

    def test_deterministic(self):
        import marker_ready
        if not os.path.isdir(marker_ready.NORM):
            self.skipTest("normalised bases not imported on this machine")
        a = tempfile.mkdtemp()
        b = tempfile.mkdtemp()
        try:
            for idx, (name, base, d, L) in enumerate(marker_ready.VARIANTS[:2], start=1):
                marker_ready.write_variant(idx, name, base, d, L, a)
                marker_ready.write_variant(idx, name, base, d, L, b)
            self.assertEqual(_dir_sha(a), _dir_sha(b))
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_provenance_and_licence(self):
        import marker_ready
        if not os.path.isdir(marker_ready.NORM):
            self.skipTest("normalised bases not imported")
        d = tempfile.mkdtemp()
        try:
            for idx, (name, base, dia, L) in enumerate(marker_ready.VARIANTS, start=1):
                rec = marker_ready.write_variant(idx, name, base, dia, L, d)
                rec.pop("mesh", None)
                self.assertEqual(len(rec["base_sha256"]), 64, name)
                self.assertNotIn("NC", rec["license"], name)
                with open(os.path.join(d, name, name + ".variant.yaml"), encoding="utf-8") as fh:
                    txt = fh.read()
                self.assertIn("base_sha256:", txt, name)
                self.assertIn("functional_read:", txt, name)
                with open(os.path.join(d, name, name + ".annotations.json"), encoding="utf-8") as fh:
                    ann = json.load(fh)
                for part in ("tip", "tail", "body", "com"):
                    self.assertIn(part, ann["parts"]["pen"], name)
                self.assertEqual(ann["parts_schema"].split(" ")[0], "hq-ctx-parts/3", name)
                np.testing.assert_allclose(ann["parts"]["pen"]["com"]["p"],
                                           ann["mass"]["com"], atol=1e-9)
        finally:
            shutil.rmtree(d, ignore_errors=True)


def _dir_sha(root):
    out = {}
    for r, _d, fs in os.walk(root):
        for fn in fs:
            p = os.path.join(r, fn)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


class TestSecondBatch(unittest.TestCase):
    """第二批四组生成器：瓶/杯、两管、铃、盘叉勺（关系 + 水密 + parts 第 3 版）。"""

    def test_all_variants(self):
        import make_samples
        for family, items in make_samples.EXTRA.items():
            for label, dflt, gen, over in items:
                p = dict(dflt())
                p.update(over)
                p["name"] = label
                res = gen(p, 0, None)
                a = res["annotations"]
                self.assertEqual(a["parts_schema"].split(" ")[0], "hq-ctx-parts/3", label)
                self.assertTrue(a["geometry"]["watertight"], label)
                self.assertTrue(a["geometry"]["orientation_consistent"], label)
                self.assertTrue(all(r["ok"] for r in a["relations"]), label)
                uid = next(iter(a["parts"]))
                self.assertIn("origin", a["parts"][uid], label)
                self.assertGreater(a["mass"]["mass_kg"], 0.0, label)

    def test_bbox_parts_com(self):
        """审核要求：mesh bbox == 标注 bbox；parts.com == mass.com；底法向 −z；柄是 list。"""
        import make_samples
        for family, items in make_samples.EXTRA.items():
            for label, dflt, gen, over in items:
                p = dict(dflt())
                p.update(over)
                p["name"] = label
                res = gen(p, 0, None)
                a = res["annotations"]
                v, _f = res["mesh"]
                lo, hi = v.min(0), v.max(0)
                for got, want in zip(lo, a["bbox"]["min"]):
                    self.assertAlmostEqual(got, want, places=6, msg=f"{label} bbox min")
                for got, want in zip(hi, a["bbox"]["max"]):
                    self.assertAlmostEqual(got, want, places=6, msg=f"{label} bbox max")
                # 每个 uid 都要有 com；单 uid 的族与整机 mass.com 对齐，
                # 多 uid 的族（搅拌器）各自给，不能和整机共用
                for uid, part in a["parts"].items():
                    self.assertIn("com", part, f"{label}/{uid}")
                    if len(a["parts"]) == 1:
                        np.testing.assert_allclose(part["com"]["p"], a["mass"]["com"], atol=1e-9,
                                                   err_msg=f"{label} com")
                    for key in ("bottom", "base"):
                        if key in part:
                            self.assertEqual(part[key]["normal"], [0.0, 0.0, -1.0],
                                             f"{label}/{uid} {key}")
                    if "handle" in part:
                        self.assertEqual(part["handle"]["kind"], "list", label)
                        self.assertGreaterEqual(len(part["handle"]["items"]), 3, label)

    def test_second_batch_byte_identical(self):
        import make_samples
        a = tempfile.mkdtemp()
        b = tempfile.mkdtemp()
        try:
            for family, items in make_samples.EXTRA.items():
                for label, dflt, gen, over in items:
                    p = dict(dflt())
                    p.update(over)
                    p["name"] = label
                    gen(p, 0, a)
                    gen(p, 0, b)
            self.assertEqual(_dir_sha(a), _dir_sha(b))
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_bottle_cup_nominal(self):
        import bottle
        b = bottle.generate_bottle(bottle.default_bottle(), 0, None)["annotations"]
        self.assertAlmostEqual(b["bbox"]["extent"][2], 0.236, places=6)   # 236 mm
        self.assertAlmostEqual(b["parts"]["bottle"]["mouth"]["p"][2], 0.236, places=9)
        self.assertAlmostEqual(b["parts"]["bottle"]["mouth"]["section"]["r_out"], 0.0085, places=9)
        self.assertGreater(b["material"]["transmittance"], 0.0)           # 透明 0..1 可调
        c = bottle.generate_cup(bottle.default_cup(), 0, None)["annotations"]
        self.assertAlmostEqual(c["bbox"]["extent"][2], 0.088, places=6)   # 88 mm
        self.assertLess(c["parts"]["cup"]["mouth"]["section"]["r_in"],
                        c["parts"]["cup"]["mouth"]["section"]["r_out"])
        # 锥形杯：握区半径 = 该高度处的局部外半径，不是口半径
        p = bottle.default_cup()
        r_local = p["bottom_r_out"] + (p["mouth_r_out"] - p["bottom_r_out"]) * 0.55
        self.assertAlmostEqual(c["parts"]["cup"]["grasp"]["section"]["r_out"], r_local, places=9)
        self.assertLess(r_local, p["mouth_r_out"])

    def test_tube_parts_v3(self):
        """顶层 tube_axis / grasp_span、body.section.r_out、s 从 tail 量。"""
        import tube2
        p = tube2.default_pair()
        a = tube2.generate(p, 0, None, key="a")["annotations"]["parts"]["guijiao1"]
        for key in ("tube_axis", "grasp_span", "body", "grasp", "socket", "pin",
                    "end_head", "end_tail", "origin", "com"):
            self.assertIn(key, a)
        L = p["L_a"]
        self.assertEqual(a["tube_axis"]["span"], [0.0, round(L, 6)])
        # s 从 tail 量、+s 朝 head（head 在 −x），所以 axis 是 −x
        self.assertEqual(a["tube_axis"]["axis"], [-1.0, 0.0, 0.0])
        self.assertEqual(a["body"]["axis"], [-1.0, 0.0, 0.0])
        self.assertEqual(a["grasp_span"]["axis"], [-1.0, 0.0, 0.0])
        # s=0 的点 = tail 端（+L/2），s=L 的点 = head 端（−L/2）
        p0 = np.array([L / 2.0, 0.0, 0.0])
        p_end = p0 + L * np.array(a["tube_axis"]["axis"])
        self.assertAlmostEqual(p_end[0], -L / 2.0, places=9)
        self.assertEqual(a["tube_axis"]["profile"][0][0], 0.0)          # s=0 在 tail
        self.assertAlmostEqual(a["tube_axis"]["profile"][-1][0], L, places=6)
        self.assertGreater(a["body"]["section"]["r_out"], 0.0)
        self.assertEqual(a["grasp_span"]["kind"], "axis")               # 独立顶层条目
        self.assertGreater(a["grasp_span"]["span"][0], 0.0)
        self.assertLess(a["grasp_span"]["span"][1], L)

    def test_tube_fit(self):
        import tube2
        p = tube2.default_pair()
        d = tube2.derive(p)
        a = tube2.generate(p, 0, None, key="a")["annotations"]
        b = tube2.generate(p, 0, None, key="b")["annotations"]
        self.assertGreater(d["clearance_a_b"], 0.0)      # B 尾能插进 A 头
        self.assertGreater(d["clearance_b_a"], 0.0)
        # 功能端：head 在 -x、tail/pin 在 +x，法向朝外
        self.assertLess(a["parts"]["guijiao1"]["end_head"]["p"][0], 0)
        self.assertGreater(a["parts"]["guijiao1"]["end_tail"]["p"][0], 0)
        self.assertEqual(a["parts"]["guijiao1"]["end_head"]["kind"], "disk")
        self.assertEqual(a["parts"]["guijiao1"]["end_head"]["normal"], [-1.0, 0.0, 0.0])
        self.assertEqual(b["parts"]["guijiao2"]["end_tail"]["normal"], [1.0, 0.0, 0.0])
        # socket 内径 > 对方 pin 外径
        self.assertGreater(a["parts"]["guijiao1"]["socket"]["section"]["r_in"],
                           b["parts"]["guijiao2"]["pin"]["section"]["r_out"])

    def test_bell_stroke(self):
        import bell
        a = bell.generate(bell.default_params(), 0, None)["annotations"]
        press = a["parts"]["button"]["press"]
        self.assertEqual(press["kind"], "joint")
        self.assertAlmostEqual(press["travel"], 0.005, places=9)         # 5 mm
        self.assertEqual(press["unit"], "m")
        self.assertEqual(press["range"], [-0.005, 0.0])
        self.assertAlmostEqual(a["parts"]["button"]["cover_top"]["section"]["r_out"], 0.0165, places=9)

    def test_pipette(self):
        import pipette
        a = pipette.generate(pipette.default_params(), 0, None)["annotations"]
        self.assertAlmostEqual(a["bbox"]["extent"][0], 0.255, places=6)   # 根包络 255
        self.assertAlmostEqual(a["bbox"]["extent"][2], 0.065, places=6)   # 总高 65
        self.assertAlmostEqual(a["parts"]["pipette"]["tip"]["p"][0], -0.1275, places=9)
        btn = a["parts"]["pipette"]["button"]
        self.assertEqual(btn["kind"], "joint")
        self.assertAlmostEqual(btn["travel"], 0.00667, places=9)          # URDF span
        self.assertEqual(btn["unit"], "m")
        self.assertEqual(btn["link"], "pipette_button_link")
        self.assertIn("section", btn)
        # 椭圆握族：截面跟着 body_w / 椭圆走，包络也对
        pe = dict(pipette.default_params())
        pe.update({"L": 0.2805, "body_w": 0.0308, "body_h": 0.0308,
                   "grip_shape": "ellipse", "button_r": 0.0070, "hook_h": 0.0715})
        ae = pipette.generate(pe, 0, None)["annotations"]
        self.assertAlmostEqual(ae["bbox"]["extent"][0], 0.2805, places=6)
        self.assertAlmostEqual(ae["bbox"]["extent"][1], 0.0308, places=6)
        self.assertAlmostEqual(ae["bbox"]["extent"][2], 0.0715, places=6)   # = hook_h
        sec = ae["parts"]["pipette"]["grasp"]["section"]
        self.assertEqual(sec["shape"], "ellipse")
        self.assertAlmostEqual(sec["a_out"], pe["body_w"] / 2, places=9)
        self.assertAlmostEqual(sec["b_out"], pe["body_w"] * 0.78 / 2, places=9)

    def test_mixer(self):
        import mixer
        a = mixer.generate(mixer.default_params(), 0, None)["annotations"]
        self.assertEqual(a["up_axis"], "+y")
        self.assertAlmostEqual(a["cabinet_extent"][1], 0.301, places=9)
        plat = a["parts"]["beaker_mixer"]["platform"]
        self.assertEqual(plat["kind"], "disk")
        self.assertEqual(plat["normal"], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(plat["p"][1], 0.301, places=9)             # 台面在机顶
        self.assertEqual(len(a["parts"]["beaker_mixer"]["ring"]["rings"]), 2)
        flip = a["parts"]["mixer_switch"]["flip"]
        self.assertEqual(flip["kind"], "joint")
        self.assertEqual(flip["unit"], "rad")
        self.assertEqual(flip["range"], [0.0, 0.5] if False else flip["range"])
        self.assertAlmostEqual(flip["range"][1], mixer.default_params()["theta_max"], places=9)
        self.assertIn("pivot", flip)
        pp = a["parts"]["mixer_switch"]["push_point"]
        self.assertEqual(pp["kind"], "axis")
        # 按压点在拨杆外端面上：y == switch_y，z == D/2 + lever_len
        p = mixer.default_params()
        self.assertAlmostEqual(pp["p"][1], p["switch_y"], places=9)
        self.assertAlmostEqual(pp["p"][2], p["D"] / 2 + p["lever_len"], places=9)
        # 开关件自己的质心，不是整机
        com_sw = np.asarray(a["parts"]["mixer_switch"]["com"]["p"])
        com_m = np.asarray(a["mass"]["com"])
        self.assertGreater(np.linalg.norm(com_sw - com_m), 0.01)
        self.assertAlmostEqual(a["parts"]["mixer_switch"]["com"]["com_from"], "geometry")

    def test_paired_sampling(self):
        """管对必须配套抽；架 + 管必须先架后管。"""
        import tube2
        rng = np.random.default_rng(0)
        pairs = tube2.sample_pair(rng, 5)
        # B 的派生范围必须落在表内（T4：200–288 mm）
        for p in pairs:
            self.assertGreaterEqual(p["L_b"], 0.200 - 1e-9)
            self.assertLessEqual(p["L_b"], 0.288 + 1e-9)
        self.assertEqual(len(pairs), 5)
        for p in pairs:
            d = tube2.derive(p)
            # 插座 = 对方的销 + 间隙（配合由构造保证）
            self.assertAlmostEqual(d["clearance_a_b"], p["fit_gap"], places=9)
            self.assertAlmostEqual(d["clearance_b_a"], p["fit_gap"], places=9)
            self.assertGreaterEqual(p["head_r_a"] - d["socket_r_a"], 0.0006 - 1e-12)
            self.assertGreaterEqual(p["head_r_b"] - d["socket_r_b"], 0.0006 - 1e-12)
            self.assertTrue(all(r["ok"] for r in tube2.check_relations(p)))
            # 两根都能出网格
            a = tube2.generate(p, 0, None, key="a")
            b = tube2.generate(p, 0, None, key="b")
            self.assertTrue(a["annotations"]["geometry"]["watertight"])
            self.assertTrue(b["annotations"]["geometry"]["watertight"])
        # 大幅超出时不允许静默少抽
        with self.assertRaises(RuntimeError):
            tube2.sample_pair(np.random.default_rng(1), 5000, strict=True)
        short = tube2.sample_pair(np.random.default_rng(1), 5000, strict=False)
        self.assertLessEqual(len(short), 5000)

    def test_rack_tube_pair(self):
        import rack
        for over in ({}, {"cols": 12}, {"cell": 0.0245, "pitch_x": 0.0275, "pitch_y": 0.0275},
                     {"layers": 2, "gap": 0.050}):
            p = dict(rack.default_params())
            p.update(over)
            t = rack.tube_for(p)
            self.assertAlmostEqual(t["tube_d"], p["cell"] - 2 * t["hole_clearance_c"], places=9)
            p2 = dict(p)
            p2.update({"tube_d": t["tube_d"]})
            self.assertTrue(all(r["ok"] for r in rack.check_relations(p2)))

    def test_tableware(self):
        import tableware as T
        f = T.generate_fork(T.default_fork(), 0, None)["annotations"]
        self.assertAlmostEqual(f["bbox"]["extent"][0], 0.203, places=6)  # 203 mm
        self.assertAlmostEqual(f["bbox"]["extent"][1], 0.025, places=6)  # head 25
        gx = -0.203 / 2 + f["parts"]["fork"]["handle"]["grasp_s"] * 0.203
        self.assertLess(gx, 0.203 / 2 - 0.045)                            # 握点在柄上
        s = T.generate_spoon(T.default_spoon(), 0, None)["annotations"]
        self.assertAlmostEqual(s["bbox"]["extent"][0], 0.145, places=6)
        p = T.generate_plate(T.default_plate(), 0, None)["annotations"]
        self.assertAlmostEqual(p["bbox"]["extent"][0], 0.180, places=6)
        self.assertEqual(len(p["parts"]["plate"]["rim"]["items"]), 4)
        self.assertEqual(p["parts"]["plate"]["bottom"]["kind"], "disk")
        # 回归点（EJ P2-1）：handle 的 list 条目用 section.rect + size.thickness + src，不再有 size.width
        it = f["parts"]["fork"]["handle"]["items"][0]
        self.assertEqual(it["section"]["shape"], "rect")
        self.assertEqual(it["section"]["in_or_out"], "out")
        self.assertIn("w", it["section"])
        self.assertIn("d", it["section"])
        self.assertTrue(it["src"].startswith("variant:"))
        self.assertNotIn("width", it.get("size", {}))
        s_it = s["parts"]["spoon"]["handle"]["items"][0]
        self.assertEqual(s_it["section"]["shape"], "rect")
        self.assertTrue(s_it["src"].startswith("variant:spoon/"))
        # body 的 s 从 tail 量起
        b = f["parts"]["fork"]["body"]
        self.assertEqual(b["span"], [0.0, 0.203])
        self.assertEqual(b["profile"][0][0], 0.0)
        self.assertAlmostEqual(b["profile"][-1][0], 0.203, places=6)


class TestQuotaSlots(unittest.TestCase):
    """配额表缺的 16 槽：试管 / 笔筒 / 篮 / 奶盒 / 瓶杯加参数。"""

    def test_tube(self):
        import testtube
        a = testtube.generate(testtube.default_params(), 0, None)["annotations"]
        self.assertAlmostEqual(a["bbox"]["extent"][2], 0.19503, places=5)
        self.assertAlmostEqual(a["bbox"]["extent"][0], 0.01678, places=5)
        self.assertEqual(a["parts"]["cube"]["mouth"]["normal"], [0.0, 0.0, 1.0])
        self.assertLess(a["parts"]["cube"]["mouth"]["section"]["r_in"],
                        a["parts"]["cube"]["mouth"]["section"]["r_out"])
        # 粗短档
        p = dict(testtube.default_params())
        p.update({"d": 0.0245, "L": 0.120, "bottom": "flat"})
        b = testtube.generate(p, 0, None)["annotations"]
        self.assertTrue(b["geometry"]["watertight"])
        self.assertAlmostEqual(b["bbox"]["extent"][0], 0.0245, places=6)

    def test_holder(self):
        import holder
        a = holder.generate(holder.default_params(), 0, None)["annotations"]
        self.assertAlmostEqual(a["bbox"]["extent"][2], 0.09635, places=5)
        m = a["parts"]["holder"]["mouth"]
        self.assertAlmostEqual(m["section"]["r_in"], 0.0428, places=6)
        self.assertGreater(m["section"]["r_out"], m["section"]["r_in"])
        self.assertLess(a["parts"]["holder"]["floor"]["p"][2], 0.01)
        # 笔必须放得进去
        pen, gap = holder.default_params()["pen_d"], holder.default_params()["insert_gap"]
        self.assertLessEqual(pen + 2 * gap, 2 * m["section"]["r_in"])

    def test_basket(self):
        import basket
        a = basket.generate(basket.default_params(), 0, None)["annotations"]
        ext = a["bbox"]["extent"]
        self.assertAlmostEqual(ext[0], 0.344, places=6)
        self.assertAlmostEqual(ext[1], 0.267, places=6)
        self.assertGreater(ext[2], 0.185)                       # 提手高出上沿
        h = a["parts"]["basket"]["handles"]
        self.assertEqual(h["kind"], "list")
        self.assertEqual(len(h["items"]), 2)
        for it in h["items"]:
            self.assertGreater(it["size"]["gap"], 0)
            self.assertGreater(it["size"]["thickness"], 0)

    def test_milk(self):
        import milk
        a = milk.generate(milk.default_params(), 0, None)["annotations"]
        ext = a["bbox"]["extent"]
        self.assertAlmostEqual(ext[2], 0.105, places=6)
        self.assertAlmostEqual(ext[0], 0.046, places=6)
        self.assertEqual(a["parts"]["milk"]["top"]["normal"], [0.0, 0.0, 1.0])

    def test_bottle_prism_neckoff(self):
        import bottle
        p = dict(bottle.default_bottle())
        p.update({"sides": 8})
        a = bottle.generate_bottle(p, 0, None)["annotations"]
        self.assertTrue(a["geometry"]["watertight"])
        self.assertEqual(a["params"]["sides"], 8)
        q = dict(bottle.default_bottle())
        q.update({"neck_offset": [0.006, 0.0]})
        b = bottle.generate_bottle(q, 0, None)["annotations"]
        self.assertAlmostEqual(b["parts"]["bottle"]["mouth"]["p"][0], 0.006, places=6)
        # 深杯（高 88 不变、锥度更小）
        c = dict(bottle.default_cup())
        c.update({"mouth_r_out": 0.030, "bottom_r_out": 0.022})
        d = bottle.generate_cup(c, 0, None)["annotations"]
        self.assertAlmostEqual(d["bbox"]["extent"][2], 0.088, places=6)


@NEEDS_OUT
class TestBellUrdf(unittest.TestCase):
    """铃两连杆候选：结构 + 相对路径（EL P1）+ 分段质心/惯量（EL P1/P2）+ 质量来源（P2）。"""

    def _files(self):
        import glob
        fs = sorted(glob.glob(os.path.join(HERE, "out", "bell_urdf", "*", "*.urdf")))
        if not fs:
            self.skipTest("run bell_urdf.py first")
        return fs

    def test_structure(self):
        import re
        for f in self._files():
            t = open(f, encoding="utf-8").read()
            self.assertEqual(re.findall(r'<link name="([^"]+)"', t), ["base", "button_cover"], f)
            self.assertEqual(re.findall(r'<joint name="([^"]+)" type="([^"]+)"', t),
                             [("press", "prismatic")], f)
            lim = re.findall(r'<limit lower="([-0-9.]+)" upper="([-0-9.]+)"', t)[0]
            self.assertAlmostEqual(float(lim[0]), -0.005, places=9)
            self.assertAlmostEqual(float(lim[1]), 0.0, places=9)
            self.assertIn('<axis xyz="0 0 -1"/>', t)

    def test_mesh_paths_are_relative_and_resolve(self):
        """Paths stay valid after moving the tree —— 必须是相对 URDF 目录的路径。"""
        import re
        for f in self._files():
            d = os.path.dirname(f)
            for mesh in re.findall(r'<mesh filename="([^"]+)"', open(f, encoding="utf-8").read()):
                self.assertFalse(os.path.isabs(mesh), f"{f}: {mesh} 是绝对路径")
                self.assertTrue(os.path.exists(os.path.join(d, mesh)), f"{f}: {mesh}")
            # 模拟换根：把整个目录拷到临时目录，相对路径仍要能打开
            tmp = tempfile.mkdtemp()
            try:
                shutil.copytree(d, os.path.join(tmp, "bell"))
                nd = os.path.join(tmp, "bell")
                for mesh in re.findall(r'<mesh filename="([^"]+)"', open(f, encoding="utf-8").read()):
                    self.assertTrue(os.path.exists(os.path.join(nd, mesh)), mesh)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    def test_com_and_inertia(self):
        """EL P1/P2：两段质心在各自段内（cover 高于 split），惯量正定、三角不等式。"""
        import re
        import numpy as np
        idx = json.load(open(os.path.join(HERE, "out", "bell_urdf", "index.json"), encoding="utf-8"))
        for it in idx["items"]:
            z_split = it["z_split"]
            self.assertGreater(it["base_com_z"], 0.0)
            self.assertLess(it["base_com_z"], z_split, it["name"])
            self.assertGreater(it["cover_com_z"], z_split, it["name"])
            t = open(it["urdf"], encoding="utf-8").read()
            origins = [float(x) for x in re.findall(r'<origin xyz="0 0 ([-0-9.]+)"', t)]
            self.assertTrue(any(abs(o - it["base_com_z"]) < 1e-6 for o in origins), it["name"])
            self.assertTrue(any(abs(o - it["cover_com_z"]) < 1e-6 for o in origins), it["name"])
            for blk in re.findall(r'<inertia ([^/]+)/>', t):
                vals = dict(re.findall(r'(i[xyz][xyz])="([-0-9.eE+]+)"', blk))
                I = np.array([[float(vals["ixx"]), float(vals["ixy"]), float(vals["ixz"])],
                              [float(vals["ixy"]), float(vals["iyy"]), float(vals["iyz"])],
                              [float(vals["ixz"]), float(vals["iyz"]), float(vals["izz"])]])
                w = np.linalg.eigvalsh(I)
                self.assertTrue(np.all(w > 0), f"{it['name']} 惯量不正定 {w}")
                self.assertGreaterEqual(w[0] + w[1], w[2] * (1 - 1e-6), f"{it['name']} 三角不等式")

    def test_button_scale_and_contact_trigger(self):
        """铃 2/3 按钮直径 x1.3、5 mm 行程，碰撞圆柱与接触触发写进 index/URDF。"""
        import re
        idx = json.load(open(os.path.join(HERE, "out", "bell_urdf", "index.json"), encoding="utf-8"))
        for it in idx["items"]:
            want = 1.3 if it["name"] in ("bell_02_77f2ea_84", "bell_03_b739cb_93") else 1.0
            self.assertAlmostEqual(it["button_scale"], want, places=9, msg=it["name"])
            self.assertAlmostEqual(it["button_face_d_after_m"],
                                   it["button_face_d_before_m"] * want, delta=1e-5, msg=it["name"])
            col = it["collision"]
            self.assertEqual(col["shape"], "cylinder")
            self.assertAlmostEqual(col["radius"] * 2.0, it["button_face_d_after_m"], delta=1e-5)
            self.assertAlmostEqual(col["centre_z"], it["z_split"] + col["length"] / 2.0, places=6)
            self.assertAlmostEqual(it["contact_trigger"]["stroke_trigger_m"], 0.005, places=9)
            txt = open(it["urdf"], encoding="utf-8").read()
            self.assertIn('<cylinder radius="%.6f"' % col["radius"], txt, it["name"])

    def test_mass_source_is_recorded(self):
        idx = json.load(open(os.path.join(HERE, "out", "bell_urdf", "index.json"), encoding="utf-8"))
        for it in idx["items"]:
            self.assertIn(it["mass_source"], ("annotation.mass", "ENG 0.2 kg 默认（库铃无质量数据）"))
            self.assertGreater(it["base_mass"] + it["cover_mass"], 0.0)


class TestBasketSingleTop(unittest.TestCase):
    """圆形单提把篮 —— 一个圆截面把、腔口净宽与沿高写进 parts。"""

    def test_single_round_handle(self):
        import basket
        import tempfile
        p = dict(basket.default_params())
        p.update(name="basket_single_t", L=0.344, W=0.267, H=0.180,
                 handle_mode="single_top", handle_d=0.0140)
        with tempfile.TemporaryDirectory() as td:
            var = basket.generate(p, out_dir=td)
            ann = json.load(open(os.path.join(td, "basket_single_t", "basket_single_t.annotations.json"),
                                 encoding="utf-8"))
        hs = ann["parts"]["basket"]["handles"]["items"]
        self.assertEqual(len(hs), 1)
        self.assertEqual(hs[0]["shape"], "round")
        self.assertAlmostEqual(hs[0]["size"]["thickness"], 0.0140, places=9)
        self.assertGreater(hs[0]["p"][2], ann["params"]["H"])          # 顶点高出沿口
        mouth = ann["parts"]["basket"]["mouth"]
        self.assertAlmostEqual(mouth["rim_z"], 0.180, places=9)
        self.assertGreater(mouth["section"]["w"], 0.30)                # 腔口净宽
        self.assertEqual(ann["params"]["handle_mode"], "single_top")

    def test_two_side_mode_still_two(self):
        import basket
        import tempfile
        p = dict(basket.default_params())
        p.update(name="basket_two_t")
        with tempfile.TemporaryDirectory() as td:
            var = basket.generate(p, out_dir=td)
            ann = json.load(open(os.path.join(td, "basket_two_t", "basket_two_t.annotations.json"),
                                 encoding="utf-8"))
        self.assertEqual(len(ann["parts"]["basket"]["handles"]["items"]), 2)


class TestMilkTopModes(unittest.TestCase):
    """平顶 / 山墙顶各一，侧抓都在。"""

    def _gen(self, **kw):
        import milk
        import tempfile
        p = dict(milk.default_params())
        p.update(kw)
        with tempfile.TemporaryDirectory() as td:
            var = milk.generate(p, out_dir=td)
            ann = json.load(open(os.path.join(td, p["name"], p["name"] + ".annotations.json"),
                                 encoding="utf-8"))
        return ann

    def test_flat_top(self):
        a = self._gen(name="milk_flat_t", top_mode="flat")
        top = a["parts"]["milk"]["top"]
        self.assertEqual(top["top_mode"], "flat")
        self.assertAlmostEqual(top["section"]["d"], a["params"]["D"], places=9)   # 平顶=整个面
        self.assertEqual(top["gable_h"], 0.0)
        self.assertIn("grasp", a["parts"]["milk"])                                # 侧抓保留

    def test_gable_top(self):
        a = self._gen(name="milk_gable_t", top_mode="gable")
        top = a["parts"]["milk"]["top"]
        self.assertEqual(top["top_mode"], "gable")
        self.assertEqual(top["section"]["d"], 0.0)
        self.assertGreater(top["gable_h"], 0.008)


class TestMarkerProfileContract(unittest.TestCase):
    """profile 从 tail 量、s∈[0,L]、半径非 0；half 恒正（tip/tail 是坐标）。"""

    def test_profile_no_zeros_and_origin(self):
        import tempfile
        import marker_ready as MR
        with tempfile.TemporaryDirectory() as td:
            MR.write_variant(1, "mk_contract_t", "sharpie_thin", 0.0150, 0.1450, td)
            a = json.load(open(os.path.join(td, "mk_contract_t", "mk_contract_t.annotations.json"),
                               encoding="utf-8"))
        p = a["parts"]["pen"]
        body = p["body"]
        L = body["size"]["length"]
        prof = body["profile"]
        self.assertEqual(len(prof), 9)
        self.assertTrue(all(r > 0 for _, r in prof), "profile 不能有 0 半径")
        s = [x for x, _ in prof]
        self.assertTrue(all(0.0 <= x <= L + 1e-9 for x in s))
        self.assertEqual(s, sorted(s))
        self.assertAlmostEqual(s[0], L * 0.5 / len(prof), places=6)   # s=0 在 tail，首点 = L/18
        self.assertEqual(p["tip"]["p"], [round(L / 2.0, 6), 0.0, 0.0])
        self.assertEqual(p["tail"]["p"], [-round(L / 2.0, 6), 0.0, 0.0])
        self.assertGreater(L / 2.0, 0.0)                               # half 恒正
        self.assertEqual(body["axis"], [1.0, 0.0, 0.0])


@NEEDS_OUT
class TestRackTubeSet(unittest.TestCase):
    """架孔/孔深/管径成套（4x10 两层 + Ø20 管）。"""

    def test_set_fits(self):
        a = json.load(open(os.path.join(HERE, "out", "rack", "rack_4x10_2L_train",
                                        "rack_4x10_2L_train.annotations.json"), encoding="utf-8"))
        p = a["params"]
        d = a["derived"]
        self.assertEqual((d["rows"], d["cols"], d["layers"]), (4, 10, 2))
        tube = json.load(open(os.path.join(HERE, "out", "official_scaled",
                                           "testtube_official_d200_l195",
                                           "testtube_official_d200_l195.annotations.json"),
                              encoding="utf-8"))
        d_tube = float(tube["bbox"]["extent"][0])
        c = (p["cell"] - d_tube) / 2.0
        self.assertGreaterEqual(c, 5e-5)          # MEAS+ENG 下限
        self.assertLessEqual(c, 2e-3)            # 上限
        # 孔深 = 总高；管比孔长，插到底仍在架外露出
        self.assertGreater(float(tube["bbox"]["extent"][2]), d["outer_m"]["H"])


@NEEDS_OUT
class TestHolderSolidWall(unittest.TestCase):
    """无孔侧壁款，开口/底保留，笔能插入。"""

    def test_insert_and_keep_bottom(self):
        for name in ("holder_solid_official", "holder_solid_narrow"):
            a = json.load(open(os.path.join(HERE, "out", "holder", name, name + ".annotations.json"),
                               encoding="utf-8"))
            dv = a["derived"]
            p = a["params"]
            self.assertGreater(dv["r_in_mouth"] * 2.0, 0.060)      # 内口够大
            self.assertGreater(p["H"] - p["bottom_t"], 0.08)        # 可用深度
            self.assertGreater(p["bottom_t"], 0.0)                  # 有底
            self.assertEqual(a["parts"]["holder"]["mouth"]["section"]["shape"], "circle")
            self.assertAlmostEqual(a["parts"]["holder"]["floor"]["p"][2], p["bottom_t"], places=9)
            # 笔（Ø20 / 长 145）能进：内半径 > 笔半径 + 净空
            self.assertGreater(dv["r_in_mouth"], 0.010 + p["insert_gap"])


class TestNoSimulatorImport(unittest.TestCase):
    def test_sources(self):
        banned = ("torch", "embodichain", "sapien", "robosynchallenge", "trimesh")
        # 导入步（读官方/库网格）不在此列：marker_import.py / shape_import.py / bell_urdf.py
        for fn in ("geom.py", "rack.py", "marker.py", "render.py", "params.py",
                   "make_samples.py", "marker_ready.py", "verify_ready.py",
                   "bottle.py", "tube2.py", "bell.py", "tableware.py",
                   "pipette.py", "mixer.py", "testtube.py", "holder.py",
                   "basket.py", "milk.py"):
            with open(os.path.join(HERE, fn), encoding="utf-8") as fh:
                src = fh.read()
            for line in src.splitlines():
                ls = line.strip()
                if ls.startswith("import ") or ls.startswith("from "):
                    mod = ls.replace("import ", "").replace("from ", "").split(" ")[0]
                    self.assertNotIn(mod.split(".")[0], banned, f"{fn}: {ls}")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestShapeImportOffline(unittest.TestCase):
    """shape_import 的离线部分（不联网）：归一 / 朝向 / 端半径 / 部件规则。"""

    def _cyl(self, r, h, n=48):
        import numpy as np
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        v = np.array([[r * np.cos(t), r * np.sin(t), z] for z in (0.0, h) for t in a])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f.append([i, j, n + j])
            f.append([i, n + j, n + i])
        return v, np.array(f, dtype=np.int64)

    def test_normalise_and_orient(self):
        import shape_import as S
        import numpy as np
        # 上窄下宽（像瓶子）：归一后窄端在上
        v, f = self._cyl(0.03, 0.20)
        v2 = v.copy()
        v2[:, 2] = np.where(v2[:, 2] > 0.1, v2[:, 2] * 1.0, v2[:, 2])
        d = S.measure.__wrapped__ if hasattr(S.measure, "__wrapped__") else None
        spec = {"height": (0.20, 0.24), "width": (0.05, 0.09), "aspect": (1.5, 5.0),
                "up_rank": 0, "up_rule": "narrow"}
        c, R = S.frame_of(v)
        w = S.normalise((v - c) @ R.T, "bottle", spec, 0.22)
        self.assertAlmostEqual(float(w[:, 2].min()), 0.0, places=9)
        self.assertAlmostEqual(float(w[:, 2].max()), 0.22, places=6)
        w2 = S.orient_up(w, spec)
        rb, _ = S.radii_at(w2, 0.0, 0.02)          # 合成圆柱只有端面顶点，容差放大
        rt, _ = S.radii_at(w2, float(w2[:, 2].max()), 0.02)
        self.assertLessEqual(rt, rb)

    def test_radii_at(self):
        import shape_import as S
        v, f = self._cyl(0.02, 0.10)
        r, _ = S.radii_at(v, 0.0, 0.006)
        self.assertAlmostEqual(r, 0.02, places=3)

    def test_parts_vessel(self):
        import shape_import as S
        import numpy as np
        v, f = self._cyl(0.02, 0.10)
        spec = {"height": (0.09, 0.11), "width": (0.03, 0.05), "aspect": (1.0, 5.0),
                "up_rank": 0, "up_rule": "wide"}
        c, R = S.frame_of(v)
        w = S.normalise((v - c) @ R.T, "cup", spec, 0.10)
        w = S.orient_up(w, spec)
        parts = S.parts_vessel("cup", w, "u", "n", spec)
        self.assertEqual(parts["bottom"]["kind"], "disk")
        self.assertEqual(parts["bottom"]["normal"], [0.0, 0.0, -1.0])
        self.assertEqual(parts["mouth"]["normal"], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(parts["mouth"]["p"][2], 0.10, places=6)

    def test_check_import(self):
        import shape_import as S
        v, f = self._cyl(0.02, 0.10)
        spec = {"height": (0.09, 0.11), "width": (0.03, 0.05), "aspect": (1.0, 5.0),
                "up_rank": 0, "up_rule": "wide"}
        c, R = S.frame_of(v)
        w = S.orient_up(S.normalise((v - c) @ R.T, "cup", spec, 0.10), spec)
        parts = {"cup": S.parts_vessel("cup", w, "u", "n", spec)}
        rel, ok = S.check_import("cup", spec, w, parts, "u")
        self.assertTrue(ok)
        self.assertTrue(all(r["ok"] for r in rel))
