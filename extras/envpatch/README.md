# envpatch - environment patch layer + P001 mixer switch

A way to change the simulation environment **without touching any official file**: patches work on an in-memory copy
of the gym config and on the env instance after it is built. This directory contains

- `envpatch_p001/` - patch **P001**: the `mixer_operating` mixer switch becomes a real two-position toggle (stand-alone package);
- `envpatch_api.py` - the small parts-registration interface (reference): how a patch hands its parts, joints, hulls and identity to a host runtime.

Status: **tested offline only** (unit tests below). It has not been validated in a simulator run in this form, and
nothing here is wired into the official repository.

## P001 - mixer switch as a two-position toggle

The official `beaker_mixer` is one mesh in which the switch lever and the body are fused. P001

- keeps the body as a rigid object, but swaps its mesh for a copy **without the lever** (mass, `body_scale`, 12 convex pieces and the random placement path stay as in the official config);
- adds the lever as a separate articulation `mixer_switch` (`switch_base` + `switch_lever`, revolute joint `flip`, axis = body +y, travel 0 -> theta_max, pushing into the body is positive; 1 convex hull for collision);
- makes it two-position, **no spring-back**: every control step the drive target is switched from the current angle (below `theta_mid - h` -> 0, above `theta_mid + h` -> theta_max, in between hold), hysteresis `h` = 2 degrees;
- reads theta and the "left finger x lever" contact per step, so a task criterion can use them (`switched_on`, `flip_by_left`, `flip_frac_*`, see `patches/P001_criteria.py`). The criterion itself is not decided here.

| quantity | default | basis |
|---|---|---|
| pivot | body y axis through (-105, 0, 38) mm | mesh: base top at z 38 mm, lever starts at z 40 mm |
| theta_max | 30 degrees (20-35 supported) | the lever tip sits 21 mm above the base; about 30 degrees pushes it flat. Geometric estimate, not a measurement |
| stiffness / damping | 0.1 N.m/rad / 0.005 N.m.s/rad | assumed; chosen so that the lever settles in the open position after the midpoint without bouncing |
| lever mass | 10 g | assumed |
| friction static / dynamic | 1.0 / 0.8 | assumed, to be tuned against push feel |

### It needs the official `beaker_mixer.obj` - we do not distribute the derived mesh

P001 needs a body mesh without the lever, the lever mesh, a convex hull and a URDF. These are **derived from the
official `assets/beaker_mixer/beaker_mixer.obj`** (sha256 `412ea93b4388eac9b4a0f2e8583405eab746cd76663ac7dfda4efd481a81538c`
at official commit 45d9b9c). They are **not included** in this repository. Instead they are split on the spot from your
own copy of the official mesh:

```python
from envpatch_p001.patches import P001_mixer_switch as P001
P001.ensure_assets()                    # finds <repo>/assets/beaker_mixer/beaker_mixer.obj, splits it (needs trimesh)
gym_cfg = P001.apply_cfg(gym_cfg)       # also calls ensure_assets() if the derived files are missing
```

The official mesh is located via, in order: the argument, `$P001_OFFICIAL_OBJ`, `$ROBOSYN_REPO/assets/beaker_mixer/beaker_mixer.obj`,
or the enclosing checkout (this directory lives inside the official repository). Derived files go to
`envpatch_p001/assets/mixer_switch/` (override with `$P001_ASSET_DIR`); the split is deterministic and writes a
`manifest.yaml` with the sha256 of every derived file. The cut box is in `generators/split_beaker_mixer.py`
(x < -0.110 m, z > 0.038 m, y in [-0.006, 0.007] m in the body frame after `body_scale` 0.3; 1611 faces move to the lever).
Because the derived mesh follows the licence of the official asset, we leave distribution to the official side.

### Interface (same shape as the other patches)

| function | meaning |
|---|---|
| `apply_cfg(gym_cfg, params=None)` | pure dict transform (deep copy); raises if something does not fit, never half-patches |
| `apply_env(env, params=None)` | call after **every** reset: put the root pose on the mixer, set limits/drive, clear targets, start the monitor |
| `readback(env)` / `verify(readback)` | effective values read back / list of problems (empty = ok); bad, missing, NaN or unreadable values never pass |
| `parts()`, `parts_scaled(parts, s)`, `push_point_at(parts, s, theta)` | functional parts and the push point at an episode scale / angle |
| `toggle_step`, `follow_root` | the two event functions (interval / reset) that `apply_cfg` hooks into the config |
| `smoke_checks(...)` | pure function: pass conditions for one episode (theta not constant, a left-finger contact happened, body not pushed over, speeds bounded, no bad observations) |

The official event manager resolves event functions by attribute name, so `install_event_funcs()` injects the two functions
into the `robosynchallenge.managers.events` module namespace of the running process (no file is modified).

## envpatch_api.py

A reference for the registration side: `register(patch_id, version, task, provider, ...)`, `collect(...)`, `augment(snapshot, ...)`.
A provider returns parts, identity, state plan, frames, joints and hulls; failures become explicit "missing" records
instead of silently disappearing. It was written for our own host runtime (a snapshot object with `.parts`, `.identity`,
`.joints` ...), so it is provided as an interface reference and has no standalone test.

## Dependencies

- Tests: Python 3.12 (tested), `numpy`, `pyyaml`, and `trimesh` (for the on-the-spot split in the asset test). No simulator, torch or GPU.
- Real runs additionally need the official simulator stack (`torch`, `robosynchallenge`).
- Needs the official repository checkout (the tests read `configs/mixer_operating/random/gym_config.json` and `assets/beaker_mixer/beaker_mixer.obj`; point `ROBOSYN_REPO` elsewhere if needed).

## Run the tests

```bash
cd extras/envpatch/envpatch_p001
python3 tests/run_all.py            # [--official-config <gym_config.json>]
```

15 checks: apply_cfg, readback/verify, rejection of bad readings, wrapper state, two-stage drive, criteria, smoke conditions,
latch dynamics, event-function injection, parts / scaled parts / push point (5 checks), and asset self-consistency
(splits the official mesh into a temp dir and checks hashes, bounding boxes and face counts).

## Known limits and open questions

1. Not run in a simulator in this form; everything above is offline-verified only.
2. The collision checker of the host may not support articulated bodies inside a scene; if it is on, the new articulation can be treated as a missing entry.
3. Whether the contact sensor accepts the new articulation (`articulation_cfg_list` entry for `switch_lever`) must be checked in a real run (read `readback["lever_contact_impulse"]`).
4. The cut leaves an opening in the body mesh (the source mesh itself has many open edges, so a global hole fill is not possible); the lever hides it, but one rendered episode should be checked by eye.
5. The two-stage drive is switched per control step (25 Hz, 4 physics substeps), not per physics substep. If the lever jitters near the midpoint, widen the hysteresis first.
6. Friction, stiffness, damping and mass are assumptions (table above).

Not included on purpose: the host-side runtime helpers (`patch_runtime`, `parts_runtime`), our run-provenance header module and
the simulator smoke script (both depend on internal tooling), and the internal registry entry and version logs.
