# Known issues and compute notes (community page)

This is a community page: **contributions are welcome, open a PR to add or correct an entry.**

Entries are written from the point of view of someone who used the environment and wants to save the next person some time. Nothing here is an official statement. Several entries are already fixed on `main` or in newer releases, so please check the linked issue before relying on one. Please keep entries factual: what you saw, on which versions, what you did. If you could not measure something, write "not measured" rather than a guess.

Links: official repo `https://github.com/EDEM-AI/RoboSynChallenge`, EmbodiChain `https://github.com/DexForce/EmbodiChain`.

## Entry format

Copy this block, fill it in, and add it to the right section.

```
#### <short title>
- **Symptom:** what you observed (one or two sentences)
- **Impact:** what it does to your run or your data
- **Environment/version:** RoboSynChallenge commit / EmbodiChain version / dexsim version / driver / GPU / renderer, as far as you know
- **Workaround:** what you did, or "none found"
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/<n> or https://github.com/DexForce/EmbodiChain/issues/<n>, or "none known"
- **Added by:** your handle (or "anonymous")
```

Baseline stack for section A unless an entry says otherwise: RoboSynChallenge `45d9b9c` (current `main` at the time of writing; the entries were first collected on `a485f91`, which differs only in two arm URDFs), EmbodiChain 0.2.4.post1 (`89c14da8`), dexsim 0.4.3. These entries were first collected from our own use of that stack; later `main` may differ.

---

## A. Environment and success-criterion issues

### A1. Installation and startup

#### LeRobot not pinned, hash mismatch, package site unreachable
- **Symptom:** the dependency install could fail because LeRobot was not pinned to a version, package hashes did not match, or the DexForce package site could not be opened.
- **Impact:** the environment could not be built.
- **Environment/version:** baseline stack.
- **Workaround:** none found.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/52, https://github.com/EDEM-AI/RoboSynChallenge/issues/55, https://github.com/EDEM-AI/RoboSynChallenge/issues/64 (closed); PR https://github.com/EDEM-AI/RoboSynChallenge/issues/58 (open).
- **Added by:** seed entry

#### Multi-GPU: CUDA and Vulkan use different device numbering
- **Symptom:** the physical GPU order (NVML) and the order after `CUDA_VISIBLE_DEVICES` remapping differ, so `gpu_id` can point to the wrong card. A semaphore import error can follow. In our runs, selecting a card through `CUDA_VISIBLE_DEVICES` made cross-card runs exit with return code 134.
- **Impact:** environment creation fails or runs on an unintended GPU.
- **Environment/version:** dexsim 0.4.3, multi-card machines.
- **Workaround:** on this version do not set `CUDA_VISIBLE_DEVICES`; pass the physical `gpu_id` directly and check the config. Log device UUID, sim `gpu_id` and torch device at startup. Installing a newer wheel alone did not fix it for us. See also section B for the extra GPU memory this causes on card 0.
- **Upstream link:** https://github.com/DexForce/EmbodiChain/issues/770 (closed; maintainers say fixed in 0.5.0, matching by UUID).
- **Added by:** seed entry

#### Segmentation fault at startup (hybrid renderer) or at exit (optix renderer)
- **Symptom:** a crash in `librender-hybrid` (VulkanCore) early in startup, or in `librender-optix` after output was already written. Root cause not known.
- **Impact:** startup failures give no episodes; exit crashes may still leave usable output.
- **Environment/version:** EmbodiChain 0.2.4.post1, dexsim 0.4.3, hybrid and optix renderers.
- **Workaround:** record which stage the process was in and the return code (139), flush output before exit, and judge whether the saved data is complete. Count startup retries separately. A 90 second wait was not a verified fix. We could not confirm a fixed release.
- **Upstream link:** https://github.com/DexForce/EmbodiChain/issues/702 (closed).
- **Added by:** seed entry

#### Process does not exit after the episode data is written
- **Symptom:** on some cloud machines the simulation process finished writing its data but did not exit; a wrapper timeout then killed it and the return code was recorded as 137 (or 124).
- **Impact:** if you accept only return code 0, you throw away complete episodes. On one machine, all 24 episodes in a 90 minute window ended this way (wall time exactly equal to the 3300 s timeout), and all 24 had a finalized dataset and a non-empty results file.
- **Environment/version:** 8x RTX 5090 cloud machines, drivers 595.71.05 and 580.173.02; EmbodiChain 0.2.4.post1 / dexsim 0.4.3 per our setup package.
- **Workaround:** decide completeness from the results file and the "dataset finalized" marker, not from the return code. We also stopped the process ourselves once the data was complete and verified readable; that change was reverted in one place because we had not checked results against the success criterion for it.
- **Upstream link:** none known.
- **Added by:** seed entry

#### Arm URDFs changed between `a485f91` and `45d9b9c`
- **Symptom:** the official repo updated the arm URDFs (`CobotMagicNoGripper.urdf`, `CobotMagicWithGripperV100.urdf`) in `45d9b9c`; `a485f91` has the earlier versions.
- **Impact:** if you set up from the earlier version, your arm model differs from the current `main`.
- **Environment/version:** `45d9b9c` vs. `a485f91`.
- **Workaround:** sync the two URDF files with the current `main` if you started from the earlier version.
- **Upstream link:** none known.
- **Added by:** seed entry

### A2. Reproducibility

#### Same seed does not give bit-identical results
- **Symptom:** the same seed and reset do not reproduce identical states on the version we tested. Some of this is in the closed-source physics layer.
- **Impact:** paired comparisons that assume identical starts are not exact.
- **Environment/version:** baseline stack.
- **Workaround:** pin the three version layers, config, seed and prior history; compare actual states, not only seeds. We did not discard otherwise valid episodes just because they were not bit-identical.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/75 (open).
- **Added by:** seed entry

#### Queries, registration and resets can advance physics
- **Symptom:** in 0.2.4.post1, CPU articulation reset/root updates, the robot randomizer and `register_info` can advance the simulation (a 1 ms update, one physical step, or 100 steps by default), so unrelated objects can move.
- **Impact:** hidden motion during reset, expert queries or state publishing.
- **Environment/version:** EmbodiChain 0.2.4.post1, dexsim 0.4.3.
- **Workaround:** pass `sim_update=false` where available and separate settling from the episode. A later EmbodiChain release changes this; check a 0-duration publish on dexsim 0.4.3 before relying on it.
- **Upstream link:** https://github.com/DexForce/EmbodiChain/issues/735 (PR, merged).
- **Added by:** seed entry

#### Drawer starts open, or opens by itself
- **Symptom:** in `drawer_open_place` the drawer can start open and keep opening with no one pulling it. The official reset check covers the initial position, not the later motion. An earlier note of ours also recorded further opening of more than 100 mm within the first 30 steps, before any robot contact (observed in earlier runs, not re-verified).
- **Impact:** episodes begin from an illegitimate state.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`), EmbodiChain 0.2.4.post1.
- **Workaround:** monitor the initial and pre-contact joint position and the first contact step; exclude affected episodes from training.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/80 (closed, partial fix).
- **Added by:** seed entry

#### Right rod roll carries over from the previous episode
- **Symptom:** in `item_assembly` the right rod keeps its roll after a reset from an unstable pose.
- **Impact:** start state depends on the previous episode.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** compare against a same-seed reference with matching history and log the deviation. The fix is for that random environment only, not a general guarantee.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/81 (closed).
- **Added by:** seed entry

#### Fork/spoon spawns inside or above the cabinet
- **Symptom:** in `drawer_open_place` the 0.4 m avoid constraint is hard to satisfy; after 100 resamples the last candidate is used.
- **Impact:** the drawer can be blocked from the start.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** check geometry in cabinet coordinates and exclude bad starts.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/82 (open), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/83 (open).
- **Added by:** seed entry

#### Mixer distractor avoidance cannot be satisfied
- **Symptom:** in `mixer_operating`, `min_distance_to_avoid` is 1.5 m, larger than the sampling box can satisfy, so sampling falls back after exhausting tries.
- **Impact:** distractors may overlap or block the task.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** check the largest reachable distance and the actual minimum separation; exclude real overlaps.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/88 (open), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/89 (open).
- **Added by:** seed entry

### A3. Reading the success criterion

#### `compute_task_state` always returns success 0
- **Symptom:** the returned success value is 0 by design; the task itself uses `is_task_success`.
- **Impact:** reading that return value makes every episode look like a failure.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`), all tasks.
- **Workaround:** record `is_task_success` and its atomic terms at every physical step and keep a running "ever succeeded" flag.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/60 (closed as by design), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/61 (open, adds a comment).
- **Added by:** seed entry

#### Latching differs between tasks
- **Symptom:** 7 tasks re-evaluate success every step; 3 tasks latch on history. This is not a single uniform rule.
- **Impact:** a per-step success trace means different things in different tasks.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** recompute from each task's own source; do not force one latch policy.
- **Upstream link:** none known.
- **Added by:** seed entry

#### `sample_loading`: bottom check uses unscaled mesh
- **Symptom:** `bottom_ok` uses vertices from `get_vertices()`, which are unscaled by default, while the tube is scaled 1.1/1.1/1.3.
- **Impact:** the bottom-alignment result can differ from the scaled geometry. It is one success branch, not the only one.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`); the default is also unscaled in 0.2.4.post1 and 0.3.2 as far as we checked.
- **Workaround:** recompute with `get_vertices(scale=True)` or rotation times (vertices times scale) plus translation, and store both results.
- **Upstream link:** none known.
- **Added by:** seed entry

#### `sample_loading`: velocity and "not held" checks pass if an API call fails
- **Symptom:** if `get_linear_velocity` or the end-effector lookup raises, the code continues with the check treated as passed.
- **Impact:** a failed check can look like success.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** recompute from real body velocity and fingertip/contact data, logging whether each term was available.
- **Upstream link:** none known.
- **Added by:** seed entry

#### `manipulate_pipette`: success does not need all conditions at the same time
- **Symptom:** the press-to-bottom count (at least 1) is kept as history, combined with the current not-failed state.
- **Impact:** a pipette press early on can count even if later poses differ.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** log the step of the first minimum reach and the tilt of both containers at that step; keep the official and our reading as separate labels.
- **Upstream link:** none known.
- **Added by:** seed entry

#### `item_assembly`: welds early, can lock a lateral offset
- **Symptom:** welding triggers at contact distance 3 mm or less and axis angle 15 degrees or less, without checking the final lateral error (2 cm). An edge name in the schedule does not match the current graph.
- **Impact:** a misaligned assembly can be locked in.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** log relative axial/lateral offset at weld time and flag lateral error over 2 cm; exclude such demonstrations.
- **Upstream link:** none known.
- **Added by:** seed entry

#### `drawer_open_place`: only tomato-to-cabinet XY distance is checked
- **Symptom:** success is `duck_drawer_dist <= 0.1`, which does not require the item to be inside the drawer, released or the drawer closed.
- **Impact:** items near the cabinet can count.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** keep the official reading and add a separate geometric label (inside the box, released, stable).
- **Upstream link:** none known.
- **Added by:** seed entry

#### `table_rearrangement`: only y and an upper z bound are checked
- **Symptom:** the fork/spoon y is within 0.02 m of plate y plus/minus 0.16 m and z minus plate z at most 0.05; there is no x, release or lower-bound check.
- **Impact:** an episode can succeed at the start or without a real place.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** flag success at step 0, success without grasp/release, and items still held or off the table.
- **Upstream link:** none known.
- **Added by:** seed entry

#### `items_handover`: height overlap uses env 0 only
- **Symptom:** with `num_envs > 1`, the overlap is computed on `_entities[0]` and broadcast.
- **Impact:** wrong values for other environments.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** `num_envs=1` avoids it; otherwise recompute per environment.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/71 (open), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/72.
- **Added by:** seed entry

#### `items_handover`: "fallen" uses the pen local x, not world z
- **Symptom:** the check uses the pen's local x axis against world z with a 75 degree limit; the axis meaning follows the asset.
- **Impact:** may not match the intended notion of "fallen" if the axis differs.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** compare angles for both the mesh axis and the semantic axis. We did not call it a bug without checking the asset axis.
- **Upstream link:** none known.
- **Added by:** seed entry

### A4. Evaluation and policy side

#### Process exits 0 but metrics file is missing
- **Symptom:** `env.close` ran before metrics were written; `truncated` returned a Python bool, so `.any` failed.
- **Impact:** a return code of 0 does not mean results exist.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`), EmbodiChain 0.2.4.post1.
- **Workaround:** check episode count, end state and that metrics exist and parse; write your own results before `close`; convert bool to tensor. Missing evidence counts as unfinished. In our runs, setting `EMBODICHAIN_SIM_EXIT_PROCESS=0` (the official ACT `eval.sh` sets it) made the metrics file appear; our own entry points all export it. We still saw a missing file on a later run.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/54 (closed), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/56 (merged).
- **Added by:** seed entry

#### SmolVLA: success is not checked inside an action chunk
- **Symptom:** a brief or early success within a chunk is not seen, and extra action steps are counted.
- **Impact:** step counts and success timing differ between our log and the adapter.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** log both readings; for training, cut at the verified completion step. Evaluation code unchanged.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/86 (open), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/87 (open).
- **Added by:** seed entry

#### SmolVLA: worker process has no seed
- **Symptom:** the worker subprocess uses its own noise stream.
- **Impact:** repeated inference on the same input differs.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** fix the worker seed and log it with an action hash; keep this separate from physics nondeterminism.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/84 (open), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/85.
- **Added by:** seed entry

#### SmolVLA: image byte type and return values
- **Symptom:** uint8 preprocessing and the three-value return were incompatible.
- **Impact:** inference errors on older code.
- **Environment/version:** older `main` before the fixes below.
- **Workaround:** use a pinned recent `main`; an open issue does not mean it is unfixed.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/53 and PR https://github.com/EDEM-AI/RoboSynChallenge/issues/57 (open); PR https://github.com/EDEM-AI/RoboSynChallenge/issues/91 and https://github.com/EDEM-AI/RoboSynChallenge/issues/92 (merged) cover the same area.
- **Added by:** seed entry

#### Expert screening checks that a plan exists, not that execution succeeds
- **Symptom:** the public evaluation/collection screening accepts a start if the planner returns a non-empty plan (IK, wrist flip and limits pass); the contact phase can still fail when executed.
- **Impact:** a start that passes screening is not proof that the task is feasible from that start.
- **Environment/version:** public evaluation code, checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** keep every candidate and record planning, execution and saving outcomes with failure reasons; do not report only the saved count.
- **Upstream link:** none known.
- **Added by:** seed entry

### A5. Simulation behavior and data

#### Top camera randomization is enabled but all ranges are 0
- **Symptom:** `look_at` only reads `eye/target/up_range`, and the current ranges are all zero.
- **Impact:** no camera variation; this can look like dropped frames.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`), EmbodiChain 0.2.4.post1.
- **Workaround:** compare real extrinsics across seeds; set non-zero ranges if you want variation.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/62 (open), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/63; https://github.com/DexForce/EmbodiChain/issues/705, https://github.com/DexForce/EmbodiChain/issues/706 (closed).
- **Added by:** seed entry

#### Lighting and materials: defaults first, then a jump every 10 steps
- **Symptom:** the first part of an episode uses default lighting, then lighting and materials change every 10 steps. With `seed=None` the interval counter continues across episodes, so the phase carries over. Explicit `reset(seed)` clears the counter, but the first trigger is still not randomized.
- **Impact:** appearance changes mid-episode; do not treat these as physical motion.
- **Environment/version:** EmbodiChain 0.2.4.post1, dexsim 0.4.3. The cross-episode counter was observed with `seed=None` only.
- **Workaround:** log the actual trigger steps and light/material hash; use a verified reset pattern.
- **Upstream link:** none known.
- **Added by:** seed entry

#### Gripper state and action use different units
- **Symptom:** in table tasks, old action was in meters (0 to 0.05) while state was 0 to 1; a newer expert sent 0 to 1 while the controller clips in meters.
- **Impact:** mixing old and new demonstrations is wrong if you go by column names.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`), and the version before the PR below.
- **Workaround:** log the policy request, controller target and real qpos with units; convert according to the actual interface.
- **Upstream link:** PR https://github.com/EDEM-AI/RoboSynChallenge/issues/78 (merged, unified the output range). Whether the whole controller chain and the drawer post-processing are consistent after that PR, we did not re-check; check current `main`.
- **Added by:** seed entry

#### `drawer_pose` / `pipette_pose` fields are all 0
- **Symptom:** a rigid-body getter is used on articulations.
- **Impact:** a 0 pose says nothing about the object.
- **Environment/version:** checked on main `45d9b9c` (same lines as `a485f91`).
- **Workaround:** mark these columns unknown and use the actual articulation or link getter.
- **Upstream link:** https://github.com/EDEM-AI/RoboSynChallenge/issues/73 (open), PR https://github.com/EDEM-AI/RoboSynChallenge/issues/74.
- **Added by:** seed entry

#### Articulation mass randomization fails; friction/damping carry over
- **Symptom:** the event calls `set_mass`, which dexsim 0.4.3 lacks, so link mass changes silently do not apply or episodes fail. Joint drive settings are not restored on reset.
- **Impact:** randomization does not do what the config says.
- **Environment/version:** EmbodiChain 0.2.4.post1, dexsim 0.4.3. Newer EmbodiChain applies mass per link; we did not run it.
- **Workaround:** disable articulation mass randomization; reset drive parameters explicitly each episode; read back parameters.
- **Upstream link:** https://github.com/DexForce/EmbodiChain/issues/218 and https://github.com/DexForce/EmbodiChain/issues/219 exist for this (we could not tell where the report for 218 originated); dexsim is closed source.
- **Added by:** seed entry

#### Partial `drive_pros` override resets a passive articulation to force drive
- **Symptom:** in 0.2.4.post1, overriding only stiffness/damping loses the default `drive_type=none`.
- **Impact:** drawers/pipettes may be driven when they should be passive. We did not test whether it explains the self-opening drawer above.
- **Environment/version:** EmbodiChain 0.2.4.post1.
- **Workaround:** set `drive_type=none` explicitly in config; read native drive type and target per joint. Robots should stay force-driven.
- **Upstream link:** https://github.com/DexForce/EmbodiChain/issues/473 (PR, merged).
- **Added by:** seed entry

#### CPU rigid-body group angular velocity is actually linear velocity
- **Symptom:** the CPU branch of `RigidBodyGroupData.ang_vel` calls `get_linear_velocity` in 0.2.4.post1.
- **Impact:** wrong values in settling or self-motion checks.
- **Environment/version:** EmbodiChain 0.2.4.post1, CPU device.
- **Workaround:** do not use that field on CPU groups; compare with finite differences.
- **Upstream link:** https://github.com/DexForce/EmbodiChain/issues/470 (PR, merged).
- **Added by:** seed entry

#### Multi-environment `register_info` joint indexing
- **Symptom:** with `num_envs > 1`, N different from the joint count raises an error; when equal, it may silently take the diagonal. Static poses may not broadcast.
- **Impact:** wrong joint values per environment.
- **Environment/version:** EmbodiChain 0.2.4.post1.
- **Workaround:** compare per-environment FK/pose with a scalar reference using 1, 2 and 6 environments.
- **Upstream link:** https://github.com/DexForce/EmbodiChain/issues/717 (closed), PR https://github.com/DexForce/EmbodiChain/issues/718 (merged).
- **Added by:** seed entry

#### Mass and fly-away: sudden jumps of the milk carton / basket (cause not isolated)
- **Symptom:** our notes record a basket displacement of about 0.7 m and one basket lift with a 327 mm displacement where the robot was in contact. We did not isolate whether this is collision solving, aggressive actions, or something else. From the config we read: test-tube rack 5.0 kg with a single convex hull; basket and milk carton 0.025 kg. We only ask whether those values are intended; we have no result that links them to the jumps.
- **Impact:** objects can leave the table or jump during contact, which makes the episode unusable and is easy to mistake for a policy failure.
- **Environment/version:** `handle_basket` and related tasks; EmbodiChain 0.2.4.post1, dexsim 0.4.3. The 0.7 m reading is from earlier notes; the 327 mm reading is from a later run.
- **Workaround:** log per-step object velocity and contact pairs and keep a separate label for "contact failure" versus "suspected physics anomaly"; an object that was touched is not "self-moving". We found no evidence that any fix exists, and nothing here says one is missing.
- **Upstream link:** none known.
- **Added by:** seed entry

---

## B. Compute and performance (main focus of this page)

Rules for this section: every number has a source in our own run logs or setup notes, and a parenthesis with machine type and versions. Anything we did not measure says "not measured". We did not estimate or round to fill gaps.

How to read the parentheses:
- **Stack** = EmbodiChain 0.2.4.post1, dexsim 0.4.3, renderer = official default (hybrid), simulation device CPU. Early single-machine measurements used `a485f91` and EmbodiChain `89c14da8` / dexsim DEVHASH `1d9ca315d`. On the later 5090 machines we used the same setup package with torch 2.7.1+cu128 and RoboSynChallenge `45d9b9c`; the exact dexsim and EmbodiChain versions were not re-read in the measurement notes for each machine, and the renderer was not recorded per run.
- Cores are the container's cgroup quota (`cpu.max`), not `nproc`: inside our containers `nproc` reported the host's core count (for example 128 where the quota was 14).
- "Episode" times include the scene build and the recorder unless said otherwise.
### B1. Single process on one card

| Reading | Machine and versions |
|---|---|
| 203 episodes/hour excluding environment build; 168 including it; median 16 s per episode (assembly, 3 cameras, recorder, strict criterion, 10 episodes) | L40S, 24-core quota, driver 580.126.09; Stack |
| 17 s per episode for the official entry; 17 to 30 s for our expert version (task not recorded in the note) | L40, 14-core quota, driver 580.126.09; Stack |
| 31.4 steps/s (click_bell, no policy model) ; 41 s to build the environment | L40, 14-core quota, driver 580.126.09; Stack |
| 12.4 steps/s with a policy model in the loop | L40, 14-core quota, driver 580.126.09; Stack |
| 14.3 steps/s (basket, with recording, 4 threads, one other expert process on the same machine) | L40, 14-core quota, driver 580.126.09; Stack |
| 38.4 s to build the environment; 350 steps in 7.1 to 7.6 s (items_handover, no cameras, 6 threads, no other load) | Tesla T4, 8-core quota, driver 535.183.06; Stack (the same task with cameras crashes, see B4) |
| 78 s for a full bell episode on an empty card, single process; 143 s for the first run | RTX 5090 x4, driver 610.57.04; Stack with torch 2.7.1+cu128 |
| 0.485 to 0.640 s per step to read two flange poses with `robot.get_link_pose`; about 12 ms per step for six links through a read of only the needed links (our recorder); recorder total about 13 ms per step | L40, 14-core quota, driver 580.126.09; Stack |

### B2. Several processes on one card (the more you stack, the slower each gets)

| Reading | Machine and versions |
|---|---|
| Assembly: 1 process 203 episodes/hour; 2 processes (12 cores each) 48 episodes/hour in total, median 131 s per episode, GPU utilization 98%. 3 and 4 processes were stopped before finishing | L40S, 24-core quota, driver 580.126.09; Stack |
| 3 processes on one card: about 140 s per episode, against 17 s for 1 process (task not recorded in the note) | L40, 14-core quota, driver 580.126.09; Stack |
| click_bell, no policy model: 1 process 31.4 steps/s; 2 processes 36 in total (15% more); 4 processes 25; 6 processes 15; 8 processes 11. Memory 3.4 GB per process. Environment build 41 s with 1 process, 219 s with 8 | L40, 14-core quota, driver 580.126.09; Stack |
| click_bell with a policy model: 1 process 12.4 steps/s; 2 processes 3.0 each, 6 in total | L40, 14-core quota, driver 580.126.09; Stack |
| Basket with recording: 1 process 14.3 steps/s; 2 processes 11.3 in total; 3 processes 12.7 in total | L40, 14-core quota, driver 580.126.09; Stack |
| CPU-heavy expert feasibility check, median per check for 1, 2, 3, 4, 5 processes on the same machine: 0.57, 1.81, 3.49, 7.25, 8.79 s. Median execution time of the same batch: 12, 45, 70, 164, 150 s. Total throughput of 5 processes was 0.39 times that of 1 process | RTX 4090, 14-core quota, driver 580.126.09; Stack |

Where we also pinned thread counts (OpenMP, MKL, BLAS and torch to 2 threads with 4 or more processes): without pinning, each process opens threads by the host core count (128 in our containers) and 5 processes could put 640 threads on a 14-core quota. We did not measure unpinned versus pinned throughput on the same run; the table above (the 5-process row) shows the cost of the contended state.

### B3. Production throughput per card and per machine

Lanes are parallel simulation processes. "Lane cores" are cores pinned per lane. Counts of episodes include episodes that did not pass our quality gate unless said otherwise.

| Reading | Machine and versions |
|---|---|
| Pipetting: 45 episodes on one card in 2 h 36 min with 3 lanes (4 cores per lane): 5.8 episodes/hour per lane, 17 per card-hour | L40 x8 machine, 112-core quota, driver not recorded; Stack |
| Basket: 9 minutes per episode per lane, 20 episodes per card-hour with 3 lanes | L40 x8 machine, 112-core quota, driver not recorded; Stack |
| 16 episodes per card-hour (task not named in the note) | L40 x4 machine, 56-core quota, driver not recorded; Stack |
| Basket batch: 17 episodes per card-hour | L40S x4 machine, 96-core quota, driver not recorded; Stack |
| Pouring, whole machine, counted by hour of completion (failed short episodes included): 116 (24 lanes), 124 (24 lanes), 120 (32 lanes) episodes/hour, so adding lanes from 24 to 32 did not raise throughput noticeably | RTX 5090 x8, 184-core quota, 483 GB memory limit, driver 580.173.02; Stack with torch 2.7.1+cu128 |
| Pouring, median wall time per episode: 1806 s over 36 episodes (the wrapper timeout was 1800 s, so most episodes ended at the timeout) | same RTX 5090 x8 machine |
| Pouring, same machine after we stopped the process once data was complete: median 1193 s, mean 1184 s over 401 episodes | same RTX 5090 x8 machine |
| One pouring episode broken down: torch compile 8.5 min, inductor 2 min, recording 851 s, exit 4 min | same RTX 5090 x8 machine |
| The same breakdown on a 4-card L40 machine: torch compile 29 s, inductor 27 s, recording 779 s | L40 x4 machine, 56-core quota; Stack |
| Pipetting, 6 episodes with 24 simulations running: 20.6, 21.3, 28.5, 29.1, 29.4, 30.4 min per episode; 7 to 9 minutes between "dataset finalized" and the results file | RTX 5090 x8, 184-core quota, 503 GB memory, driver 595.71.05; Stack with torch 2.7.1+cu128 |
| Same machine, whole-machine episodes per hour: 18 to 31 in the first hours with 24 lanes; 17 to 21 later with 18 lanes | same RTX 5090 x8 (driver 595.71.05) |
| Same machine, scene build time per episode rose from 1251, 1293, 1348 s (three consecutive hours) to 1968 to 2147 s in the following hours, with the number of lanes unchanged; recording 4 to 14 min; exit 4 to 16 min | same RTX 5090 x8 (driver 595.71.05) |
| Same machine, while slow: CPU 86% idle (load 44), memory used 146 GB of 503 GB, cgroup out-of-memory counters 0, GPU memory 8 to 20 GB per card of 32 GB. In 90 minutes, 24 episodes ended at exactly 3300 s | same RTX 5090 x8 (driver 595.71.05) |
| Basket, 4 lanes per card on 8 cards: first 4 episodes took 22 to 23 min each; of the first 62 episodes, 25 returned 0, 31 returned 124 (timeout 1800 s) and 6 returned 139 | RTX 5090 x8, 245-core quota, 503 GB, driver 595.99.02; Stack with torch 2.7.1+cu128 |
| Same machine: cards at 98% utilization drew only 125 to 145 W; idle cards drew 51 to 58 W | same RTX 5090 x8 (driver 595.99.02) |
| Pouring before any change: median 867 s per episode over 122 episodes | RTX 5090 x8, 327-core quota, 341 GB, driver 595.104.02; Stack with torch 2.7.1+cu128 |
| Pouring before any change: median 741 s per episode over 49 episodes | RTX 5090 x4, 99-core quota, 341 GB, driver 610.57.04; Stack with torch 2.7.1+cu128 |

### B4. Cores and memory

- Cores seen (cgroup quota, not `nproc`): 14 on single-card L40 and RTX 4090 containers, 24 on single-card L40S, 56 on 4-card L40, 96 on 4-card L40S, 112 on 8-card L40, 61 to 99 on 4-card rented machines, 184 to 327 on 8-card RTX 5090 machines. These are configurations, not speed results.
- Memory limits seen: 120 GiB (single-card L40), 480 GiB (4-card), 1007 GiB (8-card L40), 341 to 503 GB (rented 5090 machines).
- Effect of core count on speed with everything else equal: not measured.
- Effect of memory size on speed: not measured. On the 5090 machine in B3 with driver 595.71.05 memory was not the limit (146 GB used of 503 GB, out-of-memory counters 0).

### B5. Compute entries in the standard format

#### Several simulation processes on one card slow each other much more than expected
- **Symptom:** on one card, adding processes lowered the total or barely raised it (see B2). The assembly run went from 203 to 48 episodes/hour with a second process; click_bell without a policy gained 15% from 2 processes and lost throughput from 4 upward.
- **Impact:** "start more workers on one card" can be slower than one worker. We saw the bottleneck in three-camera rendering waiting on other processes, not CPU, in the cases we checked (not isolated on every card type).
- **Environment/version:** L40 and L40S (driver 580.126.09), Stack.
- **Workaround:** measure 1, 2 and 3 processes on your own task first; we kept 1 process per card for policy evaluation and 3 to 4 lanes per card for plain data generation on 8-card machines, based on those measurements and on out-of-memory cases below.
- **Upstream link:** none known.
- **Added by:** seed entry

#### Each process takes extra GPU memory on card 0
- **Symptom:** every simulation process reserved a context on card 0 in addition to its own card, 432 MiB each in one measurement and 502 MiB each on another machine. Cards that had earlier held many processes showed leftover memory (13 contexts of 426 MiB = 5.7 GB on one card, 9 of 432 MiB = 4.0 GB on another) after the processes were gone.
- **Impact:** a rule like "wait until card 0 memory is under 500 MiB" never becomes true. On 32 GB cards with 4 lanes per card, card 0 reached 30.7 to 32.6 GB in one case and 29.4 of 32.6 GB in another, and CUDA out-of-memory errors followed (12 of the 25 episodes that ended with return code 134 in one hour on the machine with driver 595.99.02).
- **Environment/version:** dexsim 0.4.3, EmbodiChain 0.2.4.post1; RTX 5090 (drivers 595.99.02 and 580.173.02) and L40 machines.
- **Workaround:** select the card with the `--gpu N` option of the entry point only, and also set `GPU_PIN=N` so warp and torch compute on that card; on one machine each process still held 502 MiB on card 0 even with `GPU_PIN` set. We ran one fewer lane on card 0 on the 32 GB machines.
- **Upstream link:** https://github.com/DexForce/EmbodiChain/issues/770 for the related device numbering.
- **Added by:** seed entry

#### Vulkan ICD and missing libraries on fresh cloud images
- **Symptom:** `import dexsim` failed on missing `libGLX.so.0` or `libEGL`; `/etc/vulkan/icd.d/` was empty on one image (the NVIDIA ICD was under `/usr/share/vulkan/icd.d/`); with no ICD file set, `vulkaninfo` also listed a software `llvmpipe` device next to the real card.
- **Impact:** import errors, or a risk of rendering on the software device.
- **Environment/version:** Ubuntu 22.04 cloud images; drivers 580.126.09 and 535.183.06; dexsim 0.4.3.
- **Workaround:** install the GL/EGL packages (`libglx0 libgl1 libglvnd0 libegl1` and friends); point `VK_ICD_FILENAMES` only at the NVIDIA ICD; check `vulkaninfo --summary` shows `deviceName` and `driverName` of NVIDIA. On an 8-card RTX 5090 machine we saw eight NVIDIA devices and no software rendering when the ICD was set.
- **Upstream link:** none known.
- **Added by:** seed entry

#### T4 with driver 535 cannot create the OptiX denoiser
- **Symptom:** creating the `items_handover` environment with cameras aborted in `optixDenoiserCreate`. Turning denoising off, switching the denoiser to NRD, or the FASTRT renderer gave the same abort; RASTER failed when creating a background material. The OptiX denoiser weight file `/usr/share/nvidia/nvoptix.bin` was not in the container; a working RTX 4090 machine with driver 580 had it (60,858,204 bytes).
- **Impact:** no camera data on that machine. Without cameras the same task ran (see B1).
- **Environment/version:** Tesla T4, driver 535.183.06, EmbodiChain 0.2.4.post1, dexsim 0.4.3, renderer hybrid.
- **Workaround:** none found. We did not test copying the file from the same-version driver package.
- **Upstream link:** none known.
- **Added by:** seed entry

#### Multi-card training failed on one rented RTX 5090 machine with driver 610.57.04
- **Symptom:** a 4-card training smoke test ended with `CUDA error: an illegal memory access` on one rank and SIGABRT. The same training package ran 200 steps on an 8-card RTX 5090 machine with driver 595.104.02. Our NCCL test (all_reduce and broadcast on 4 cards) also failed intermittently on the 610 machine; the logs showed `CUDA peer access: Not supported` and the cards could not talk directly.
- **Impact:** no multi-card training on that machine.
- **Environment/version:** RTX 5090 x4, driver 610.57.04 (CUDA 13.3), torch 2.7.1+cu128, NCCL 2.26.2.
- **Workaround:** `NCCL_SHM_DISABLE=1` helped only sometimes and was not reliable. The driver is our suspected cause; we did not isolate it from sharing the card with recording jobs.
- **Upstream link:** none known.
- **Added by:** seed entry

### B6. Not measured

- Like-for-like throughput of the same task on the same config across L40, L40S, RTX 4090 and RTX 5090: not measured. The readings above come from different tasks, lane counts and quotas and must not be read as a ranking.
- Production throughput on RTX 4090 for data generation: not measured.
- Speed effect of core count (controlled) and of memory size: not measured.
- Hybrid versus optix renderer speed: not measured.
- Multi-card training throughput on L40, L40S and RTX 4090: not measured beyond one single-card pair, 36.0 samples/s on L40S and 43.2 samples/s on RTX 4090 (single card, one training run each; software stack for the training not recorded in our note).
- Real RTX 5090 versus other cards on an empty single card for the same task: only the single bell episode in B1 (78 s) exists.

---

## C. Weakly supported

We include these because they may help, but our evidence is thin; please do not rely on them without checking.

- **Drawer opening before contact** (A2): the 100 mm in the first 30 steps comes from earlier runs and was not re-verified. We did not test whether the partial `drive_pros` override explains it.
- **Mass and fly-away** (A5): the 0.7 m and 327 mm readings are single observations; the cause was not isolated; the mass values are only a question about the config.
- **Gripper units after the unifying PR** (A5): we did not re-check the full controller chain.
- **Lighting phase across episodes** (A5): observed with `seed=None` only; we did not repeat it with explicit seeds beyond reading how the counter is reset.
- **Expert screening** (A4): based on reading the screening code and on a report we could not tie to a reproduction.
- **Pipette nominal pose versus functional point:** the pipette root, the button link and the tool point can differ, and contact settling can move the root. We have no measured offset and no original reproduction, so we do not give a number.
- **Articulation mass on newer EmbodiChain:** we read that newer releases apply link mass; we did not run it.
- **SmolVLA image type:** we judge it fixed on recent `main` because an organizer PR touches the same code; we did not rerun old and new side by side.
- **Segfault causes** (A1): the cause and a fixed release are unknown; the retry advice is only what we did.
- **Slow scene build on one 5090 machine, driver 595.71.05** (B3): the jump in scene build time has no verified cause. We guessed that shader or compile caches were cold each episode, or that the host changed; we did not run a controlled test.
- **5090 slow path** (B3): the low power at 98% utilization is only a hint that rendering is on a slow path; we did not confirm which stage, and an idle-card single run on a driver 610 machine (78 s) is the only contrast we have.
- **Driver 610 and multi-card training** (B5): we suspect the driver; this is inference, not an isolated cause.
- **CPU model effect:** one note suggests a single-thread-limited CPU (an EPYC 7K62 generation) behind slow episodes on one machine, with no per-core comparison.
- **Hourly rates on 5090 machines** (B3): some counts include failed short episodes and timeouts, so they are not a measure of good data per hour.

---

Reported from our own use; not verified by the organizers
