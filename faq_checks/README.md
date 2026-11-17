# FAQ-aligned success criteria (optional)

The official success flag of each task is a compact check written for the simulator.  For several tasks it is looser than,
or different from, what the challenge FAQ describes (for example "put the basket down stably", "release the button",
"the tube is seated on the rack floor").  This package re-judges every episode **sentence by sentence against the FAQ** and
reports the result in separate columns.  It is optional, off by default, and never changes the official criteria or the
official episode outcome.

## Where the criteria come from

The criteria are the conditions of the **final-round FAQ** (item 06, "What are the real-robot success criteria for each task?", which has one paragraph per task). Each task paragraph is split into the sentences below; each sentence becomes one flag in the `faq` block. The FAQ gives **no numbers**, so every threshold in the modules (heights, distances, angles, the 20-step hold) is **tentative**: our reading of "lifted", "stably", "visibly", and so on. They are documented in each module's docstring and are easy to change.

| task | FAQ item | sentences it is split into (short FAQ quote, then our flag) |
|---|---|---|
| click_bell | 06, `Real_click_bell` | "Press the bell on the table, pressing the yellow area." → pressed; hand released; base not pushed |
| drawer_open_place | 06, `Real_drawer_open_place` | "Fully open the upper drawer on the right." → drawer fully opened, still open at end; "Place the toy tomato ... inside the open drawer." → tomato in cavity, hand off tomato |
| handle_basket | 06, `Real_handle_basket` | "Pick up the milk on the left and place it in the basket." → milk lifted, milk in basket; "Move the basket ... toward the center ... producing a clear displacement." → basket carried, basket at target; "Set the basket down stably." → basket set down |
| item_assembly | 06, `Real_item_assembly` | "Pick up the two silicone tubes from the table." → both lifted; "Connect the head of the right tube to the tail of the left tube." → head joined tail; "Once inserted, the task ends." → inserted, still joined |
| items_handover | 06, `Real_items_handover` | "Pick up the marker on the right and place it stably in the pen holder." → pen lifted, pen in holder, pen stable; "The pen holder must not fall over." → holder upright |
| manipulate_pipette | 06, `Real_manipulate_pipette` | "Pick up the pipette ..." → picked up; "Move it properly above the beaker." → above beaker; "Visibly press down the button ..., then release the button." → button pressed, button released; (our additions: still held; upright, which the official check also requires) |
| mixer_operating | 06, `Real_mixer_operating` | "Pick up the beaker on the right ..." → beaker lifted; "... place it stably in the correct position on the mixer, on the small circle ..." → on the circle, resting stably; "Successfully press the button on the left side of the mixer." → button pressed, switch on and hand off; (order: lifted < placed < pressed) |
| sample_loading | 06, `Real_sample_loading` | "Pick up the test tube on the right ..." → tube lifted; "... insert it stably into the test-tube rack." → tube in hole, tube seated, tube vertical; "It is sufficient for the tube to stay in the rack without falling." → hand off, tube still |
| table_rearrangement | 06, `Real_table_rearrangement` | "Place them stably on either side of the plate ..." → on both sides, placed stably; "... approximately parallel to each other ..." → roughly parallel; "tidy arrangement" is not a separate flag |
| water_pouring | 06, `Real_water_pouring` | "Pick up the cup and bottle from the table." → bottle lifted, cup lifted; "... perform the motion of pouring water from the bottle into the cup." → poured; "Place both the bottle and cup stably on the table." → bottle / cup set down |

Where a flag is our own addition rather than a FAQ sentence (the pipette's "still held", and its "upright", which the official check also requires), it is marked as such. The official simulator criteria are not replaced and remain the default.

## Switch

In the policy's `deploy_policy.yml` (or on the command line):

```yaml
faq_checks: true        # default: false (nothing changes when it is off)
faq_extra_steps: 40     # optional: how long an episode may run on after the official success so the 20-step hold can be observed
```

```bash
python scripts/eval_policy.py --config policy/act/deploy_policy.yml --overrides faq_checks true
```

With the switch on:

* the official outcome is latched at the first step the official check succeeds (exactly where the default loop would stop), and
  `success` / `action_steps` in `evaluation_metrics.json` are unchanged;
* the episode then keeps running for at most `faq_extra_steps` more steps (or until the FAQ criteria have held for 20 steps), so
  the "stays completed for ~20 steps" requirement can be judged;
* each entry of `episodes` gets a `faq` block (`success`, `conditions_last_step` = one flag per FAQ sentence, `hold_steps`),
  `summary` gets `faq_success_count` / `faq_success_rate`, and the console prints the FAQ column below the official one.

An episode is FAQ-successful when every sentence holds on the last step and has held for at least 20 consecutive steps.
A sentence is `true`, `false` or `null` (not enough evidence, e.g. contacts are not available); an episode with a `null` sentence is not counted as successful.

## Layout

| file | content |
|---|---|
| `<task>_faq.py` (10 files) | one independent module per task: the FAQ sentences and their numbers in the docstring, `Judge`, `make_judge(ctx)`, `obs_from_scene(scene, ctx)` |
| `common.py` | numpy helpers shared by the task modules |
| `parts.py`, `parts_default.json` | functional-part table (bottom, mouth, tip, ...) per task object at body_scale 1, and its scaling to an episode |
| `scene.py` | `FlatScene`: a simulator-independent view of one step |
| `sim_reader.py` | builds a `FlatScene` from a live EmbodiChain env (getters only) |
| `runtime.py` | `FaqChecker` (one env) and `FaqEnvProxy` (hooks `reset` / `step`) |

The task modules do not import each other.  To take only some tasks, keep `common.py`, `parts.py`, `parts_default.json`, `scene.py`,
`sim_reader.py`, `runtime.py`, `__init__.py` and the wanted `<task>_faq.py` files (and drop the other names from `TASKS` in `__init__.py`).

## What is read from the simulator

Poses of the task's rigid objects and articulations, articulation joint positions (bell button, drawer, pipette button slide), the
wrist flange of the arms where a sentence needs "in the hand", the control frequency, the table extent from the gym config, and robot-vs-object
contacts.  Contacts come from the physics scene's CPU contact buffer (the same buffer `ContactSensor` reads), switched on once with
`enable_contact_data_update_on_cpu(True)`.  Limitations:

* the buffer holds the contacts of the **last physics sub-step** of a control step;
* with GPU physics contacts are unavailable; sentences that need them come out as `null`;
* `robot.get_link_pose` is slow on CPU devices in EmbodiChain 0.2.4 (it reads every link); only click_bell, manipulate_pipette and water_pouring need a wrist pose.

## Known gaps

* `mixer_operating`: the official asset has no switch joint (the button is a contact region), so "button pressed" and "switch on, hand off" need an
  articulation `mixer_switch` with link `switch_lever`.  Without it these sentences are `null`.
* `item_assembly`: the insertion step is the step on which the env's own attach (weld) flag turns on.
* `sample_loading` speed uses the displacement per control step divided by the control period, not the simulator's own velocity reading.

## Parts table

`parts_default.json` holds, for every task object, the functional parts the sentences refer to, measured on the official asset files; the `_meta`
entry says how each entry was obtained.  `scripts/measure_faq_parts.py` reproduces the entries that are plain measurements of one mesh.

## Tests

```bash
python -m unittest discover -s tests/faq_checks -t .
```

The tests need only numpy.  They include recorded episodes (`tests/faq_checks/fixtures/`) whose expected verdicts were produced by the criteria
code these modules were rewritten from, and mocks of the EmbodiChain getters for `sim_reader.py`.
