# shape_gen - parametric and library-based object shape generators

Generators for the simulation objects used in our extra variants: test-tube racks, markers/pens, bells, plates, forks,
spoons, bottles, cups, baskets, milk cartons, pen holders, mixers, pipettes, two-part tubes and drawer handles. Every
variant is written as a mesh (OBJ + PLY), an `*.annotations.json` (functional parts: grasp points, holes, tips, mass,
inertia, placement surfaces ...) and a `variant.yaml` (generator, parameters, seed, sha256 of every file, licence).
The mesh files themselves are **not** in git, see "Assets" below.

Not an official component: nothing here changes the official repository, and nothing is wired into the official
configs. It is a toolbox to produce and check extra object variants.

## What is in this directory

| file | role |
|---|---|
| `geom.py` | mesh accumulation, watertight / orientation self-checks, mass and inertia, OBJ / PLY / JSON / YAML export |
| `params.py` | parameter ranges with their provenance tags, relation thresholds |
| `rack.py`, `testtube.py`, `tube2.py`, `marker.py`, `bell.py`, `bottle.py`, `basket.py`, `milk.py`, `holder.py`, `mixer.py`, `pipette.py`, `tableware.py` | procedural generators (numpy only; `generate(params, seed, out_dir)`; same parameters and seed give byte-identical files) |
| `pipette_reshape.py` | reshapes the official pipette per segment (reads the official mesh from your checkout, see `ROBOSYN_REPO`) |
| `drawer_handle.py` | the two drawer handle proposals (`shallow_arc`, `bridge`) as convex blocks; can assemble a full drawer from your own copy of the official drawer files |
| `milk_from_glb.py` | derives the two milk-carton meshes from the original GLB files |
| `marker_import.py`, `marker_ready.py`, `marker_41.py`, `marker_fix.py` | ready-made marker pipeline: import (PCA normalise) -> diameter bins -> mid/thick variants -> re-measured barrel |
| `shape_import.py`, `ware_import.py`, `ware_select.py`, `ware_ready.py` | library-first import pipeline for bottle / cup / basket / milk and bell / plate / fork / spoon (licence filter CC BY / CC0 only, download, normalise, annotate) |
| `verify_ready.py`, `verify_ware.py` | rebuild into a temp dir and compare sha256; print the provenance table |
| `make_samples.py`, `render.py`, `recenter_rack.py`, `bell_urdf.py`, `measure_nominal.py` | sample sheets, a small z-buffer renderer, rack re-centring, two-link bell URDFs, measurement of nominal sizes of the official assets |
| `params.py`, `selection.json`, `nominal_measured.json` | parameter tables, selection record, measured nominal sizes of official assets (numbers only) |
| `test_shape_gen.py`, `test_ware.py`, `test_drawer_handle.py` | offline unit tests |
| `WARE_README.md` | notes (Chinese) on the library-import shapes |

## Dependencies

- Tested with Python 3.12, `numpy` for the procedural generators and tests.
- `trimesh` (and `scipy` for connected components / hulls) only for the import steps, `drawer_handle.py`, `milk_from_glb.py`, `measure_nominal.py`, `pipette_reshape.py`.
- No simulator, GPU or network is needed for the tests. The import steps download from public mirrors and read local caches (`~/objaverse_marker`, `~/objaverse_ware`).
- `ROBOSYN_REPO` (environment variable) = path to an official checkout; used by `measure_nominal.py`, `pipette_reshape.py`, `marker_import.py`.

## Usage

```bash
cd extras/shape_gen
python3 make_samples.py                      # sample variants + sheets (writes ./out, ./images)
python3 -m unittest test_shape_gen test_ware test_drawer_handle -v
python3 drawer_handle.py --out handles       # handle blocks only, no official files needed
```

```python
import rack
params = dict(rack.default_params(), rows=4, cols=10)
variant = rack.generate(params, seed=0, out_dir="/some/dir")
# -> {"name", "fpath", "body_scale", "files": {name: sha256}, "source": {...}, "grasp", "status", ...}
```

## Coordinates and annotations

| family | origin | up | notes |
|---|---|---|---|
| rack | bounding-box bottom centre | +z | the official rack is z-centred; `recenter_rack.py` converts |
| marker | geometric centre | +z | main axis +x, tip at +L/2 (same convention as the official pen) |
| bell / plate | bottom centre | +z | |
| fork / spoon | bbox centre | +z | long axis +x, head towards +x |

`parts` in the annotations follows a small "ctx parts" schema (`hq-ctx-parts/3`): `kind / p / axis / size / tol / src`
per part. `functional` / `placement` / `grasp` blocks are key point-axis annotations in the style of RoboTwin-OD.

## Assets (meshes) and licences

The meshes (18 items) are included under `extras/assets/`; `extras/assets_manifest.json` lists the sha256 of every file.
Items we generated (three test-tube racks, two drawer handles, one solid-wall pen holder) are CC BY 4.0, attribution
"Yi Li (liyifreddy)". Third-party items keep their own licences (CC BY 4.0, one CC0) with per-item attribution in
`extras/assets/ATTRIBUTION.md`. The code is Apache-2.0.

## Known limits

- Tests that need generated data (`./out/...`, e.g. bell URDF index, rack/tube fit, solid-wall holder) are skipped on a clean checkout. The pure-numpy tests (parameter relations, determinism, bad-parameter rejection, annotations) always run.
- Not included: the generator for the 14 "official-size scaled" variants (derived from official assets), the internal review pages, and the staging area.
- Library-import steps need network access and the original model licences were checked at import time; the 11 third-party models under `extras/assets/` were re-checked against their pages on 2026-11-17.
- Dimensions in `params.py` carry provenance tags; where a tag mentions reference racks or markers, the numbers come from the challenge's reference material and were not measured by us.
