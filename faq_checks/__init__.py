"""Optional FAQ-aligned success criteria for the ten RoboSynChallenge tasks.

Off by default.  When enabled in ``scripts/eval_policy.py`` (``faq_checks: true``) every episode is additionally judged by the
criteria in ``<task>_faq.py``, and the results are reported next to the official success flag as separate columns.  The official
criteria and the official episode outcome are not changed.

Layout
    common.py       numpy helpers shared by all task modules (poses, tilt, hold counting, three-valued logic)
    parts.py        functional-part tables: loading, scaling to an episode's body_scale, ``EpisodeContext``
    parts_default.json   the parts at body_scale = 1 for every task object (see its ``_meta`` for provenance)
    scene.py        ``FlatScene``: simulator-independent view of one step (poses, joint positions, contacts, events)
    sim_reader.py   builds a ``FlatScene`` from a live EmbodiChain env (getters only)
    runtime.py      ``FaqChecker`` (one episode) and ``FaqEnvProxy`` (hooks reset / step of an env)
    <task>_faq.py   one independent module per task: ``Judge``, ``make_judge(ctx)``, ``obs_from_scene(scene, ctx)``

Modules for different tasks do not import each other; the shared files are ``common.py``, ``parts.py`` and ``scene.py``.
"""
import importlib

TASKS = ("click_bell", "drawer_open_place", "handle_basket", "item_assembly", "items_handover",
         "manipulate_pipette", "mixer_operating", "sample_loading", "table_rearrangement", "water_pouring")


def task_module(task):
    """Import and return ``<task>_faq``; raises KeyError for an unknown task name."""
    if task not in TASKS:
        raise KeyError("no FAQ criteria for task %r (known: %s)" % (task, ", ".join(TASKS)))
    return importlib.import_module("%s.%s_faq" % (__name__, task))
