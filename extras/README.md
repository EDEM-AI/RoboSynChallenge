# extras - optional add-on toolkit for RoboSynChallenge

Self-contained add-ons that sit **next to** the official code. This directory only adds files: no official file is
modified, nothing is imported by the official code, and nothing changes the official criteria or scoring. Take the
whole directory or just one tool; the three tools are independent of each other.

| directory | what it is | needs |
|---|---|---|
| [`env_anomaly/`](env_anomaly/README.md) | reads one recorded episode and checks it, rule by rule, against known environment bugs; separates environment errors from executor errors | numpy |
| [`envpatch/`](envpatch/README.md) | environment patch layer that never edits official files, plus patch P001: the mixer switch as a real two-position toggle | numpy; official `beaker_mixer.obj` (split on the spot); trimesh for the split |
| [`shape_gen/`](shape_gen/README.md) | parametric / library-based object shape generators (racks, markers, bells, tableware, bottles, baskets, milk cartons, holders, drawer handles ...) with annotations and provenance | numpy; trimesh for import steps |
| `assets/` + `assets_manifest.json` | the 18 extra object items (meshes, `variant.yaml`, annotations), with licence files and a sha256 manifest | none (data only) |

## Assets are included

The 18 extra object items (meshes, `variant.yaml`, annotations) are under `extras/assets/`, about 11 MB in total, with
`ATTRIBUTION.md` and `LICENSE`. `assets_manifest.json` lists every file with its sha256, source and licence (paths are
relative to `extras/assets/`).

## Licences

- **Code** (everything under `extras/` except `extras/assets/`): Apache-2.0, same as the official repository (see the top-level `LICENSE`).
- **Items we made** (three test-tube racks, two drawer handles, one solid-wall pen holder): CC BY 4.0, attribution "Yi Li (liyifreddy)".
- **Third-party items** (11 source models: bells, forks, spoons, plates, a marker, milk cartons): each under its original
  licence, CC BY 4.0 except one CC0 work; title, author, link, licence and changes are listed item by item in
  `assets/ATTRIBUTION.md` and in `assets_manifest.json`. Licences were re-checked against the original model pages on 2026-11-17.

## Status and tests

Everything was developed and verified **offline**, with synthetic cases and unit tests; none of it was run on simulator
output as part of this change. Each README lists its limits.

```bash
python3 extras/env_anomaly/tests/test_envanom.py
python3 extras/envpatch/envpatch_p001/tests/run_all.py
cd extras/shape_gen && python3 -m unittest test_shape_gen test_ware test_drawer_handle
```

Tested with Python 3.12. Contact: GitHub user `liyifreddy`.
