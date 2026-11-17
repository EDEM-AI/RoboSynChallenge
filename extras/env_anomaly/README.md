# env_anomaly - offline detection of known environment anomalies in recorded episodes

The official simulation has known bugs (drawer that opens by itself, objects that sink or fall through the table, tubes
welded too early, ...). This tool reads **one recorded episode** and gives every automatically checkable bug its own
detector: verdict + evidence window (step numbers) + measured values + rule version. It separates **environment errors**
from **executor errors** (the expert or the policy that produced the episode) and marks anything unclear as `unknown`
instead of silently excluding it.

Status: written and tested **offline only** (synthetic positive/negative episodes). It has not been run against
simulator output in this repository, and thresholds marked B below were only looked at on our own recordings.

Rule ids (`E01`, ..., `E32`, `X1`, `X2`, `CRIT.*`) are **local to this tool**; they are not official issue numbers. Where an
official issue exists it is mentioned in the table (for example #80 for the drawer that does not start closed).

## Verdicts and actions

| verdict | meaning | default action |
|---|---|---|
| `confirmed` | environment / criterion / recording error with strong evidence (grade A threshold, or grade B plus a contact column proving neither the robot nor what it holds touched the object) | `exclude` (leave out of the training view, keep the data); `flag` for opening-state and invalid-column cases |
| `candidate` | looks like an environment error but evidence is weak (usually no contact column, or only a B threshold) | `spot_check` |
| `unknown` | cannot decide (gray zone, missing field, missing reference); `missing` says what is missing | `flag` |
| `expert` | the phenomenon exists but was caused by the executor (touched it, knocked it off the table, welded early, short action table, did not settle) | `expert_fail` (counts as executor failure) |
| `clear` / `na` | checked, nothing found / not applicable | `none` |

An episode's `episode_action` is the heaviest of its results: exclude > spot_check > official_only > expert_fail > flag > none.

## Detectors

| detector | what it checks | executor's fault when | ceiling |
|---|---|---|---|
| `E01.gap` | drawer opening at start minus closed limit > 5 mm (official issue #80) | start, nothing moved yet | confirmed, flag |
| `E01.self_open` | drawer opens > 5 mm while the fingertip is far from the handle | hand within 5 cm, or robot / held object touched the drawer | with contact column and no touch: confirmed; without: candidate |
| `E02`, `E03`, `E04` | start pose of the right rod vs the batch median; distractor close to cabinet / beaker | start only | candidate |
| `CRIT.freebie`, `CRIT.official_only` | criterion disagreements (needs an external `judges` module, see below) | depends on reason | candidate; unknown without `judges` |
| `X1` | bell cover stroke (header limits) shorter than the official 4.8 mm threshold | stroke sufficient but not pressed | confirmed with limits, else candidate |
| `X2` | basket episode ends before the official 55-step count although milk is in the basket | always the action table | expert |
| `E09` | tube 2 welds to tube 1 while laterally misaligned > 2 cm | lateral > 2 cm | expert |
| `E12/E31` | num_envs > 1 | - | candidate; `na` when 1 |
| `E17` | appearance (light/material) event inside the episode | - | candidate, flag |
| `E18` | joint readings non-finite or jump > 3 cm | robot in contact | confirmed / candidate |
| `E19` / `R0` | all-zero pose, non-finite, non-increasing time | - | confirmed (flag or exclude) |
| `E21`, `E21.fall`, `E21.settling` | self-motion, velocity jumps, sinking / falling (gate 7 rules R1, R3, R4), classified by fingertip distance and contact column | contact or fingertip <= 5 cm | confirmed / candidate / unknown |
| `E22/E28` | episode file not closed (`.part`, no footer) | - | confirmed, exclude |
| `E23` | gripper command 0-1 clipped to 0.05 by a metre-based controller | - | confirmed, flag (interface unit) |
| `E30` | object moving but velocity readback is all zero | - | confirmed, flag |
| `E32` | drive stiffness / damping read-back attached | - | na |

Not auto-checkable (`detectors.NOT_AUTO`, with the missing information): E05 (cross-episode), E13 (needs functional axes),
E14, E15, E16, E20, E24, E25, E26, E27 (needs a physics step index the recorder does not store), E29.

Conventions: fingertip = both finger links when present, else flange x `header.tcp.tip`; handle = handle link when present,
else cabinet root x (0.105 + q, 0, 0.035). The contact column exists only in a subset of episodes and only for the last
physics substep, so "did not touch" is only claimed with the full window checked; episodes without it never get `confirmed`.

## Input format and the vendored gate 7

`envanom/gate7.py` is the offline implementation of the recording rules R0-R8 and the reader of our `rsrec` episode files
(`ep_*.jsonl.gz`, schema rsrec/1.2, 1.1 readable). It is **vendored here because the detectors call it** (`Ctx` builds its
per-step series with `gate7.series`, and `E17 / E19 / E21` use R0, R1, R3, R4, R8). It is not part of any other pull request.
All detectors work on one episode dict `{path, header, steps: [{t, state, contacts?, vel?, ...}], footer, plans, source}`;
build that dict yourself to feed other recordings (see `envanom/load.py`).

Optional external pieces (absent here, the tool degrades): a `judges` module for the `CRIT.*` detectors (without it they
report `unknown`), and task-specific opening checks (set `RSREC_TOOLS_ROOT`). The original reader for official evidence files
used internal tooling and is not included.

## Dependencies

Python 3.12 (tested), `numpy`. No simulator or GPU.

## Usage

```bash
cd extras/env_anomaly
python3 -B tests/test_envanom.py                   # 29 synthetic cases; add --write-result to save tests/test_envanom_result.txt
python3 -B -m envanom.run rsrec <recording dir...> --out <output dir> [--label name]
```

```python
from envanom.load import load_rsrec
from envanom.run import check_batch, train_view
recs = check_batch([load_rsrec(p) for p in episode_files])   # one batch per task (E02 needs the batch's start-pose median)
tv = train_view(recs)   # {'exclude': [...], 'spot_check': [...], 'official_only': [...], 'expert_fail': [...]}
```

Output: `episodes.jsonl`, `findings.csv`, `summary.json` / `summary.md`, `train_view.json`. Every result's `rule` carries the rule-set version
(`envanom/0.1:...`), so labels from different versions are never mixed.

## Not done / known limits

- No recordings with "no action" (hold) starts were available, so there is no negative control for pure environment drift.
- The contact column covers a minority of episodes, so most environment conclusions stop at `candidate`; record with all contacts to get more `confirmed`.
- E01 handle offset, E21 gray-zone thresholds and E02 / E03 / E04 thresholds are grade B: only looked at on our own recordings, not independently validated.
- No write-back table for spot-check outcomes; no cross-batch reference for E02.
- E27 needs the recorder to log a physics step index.
