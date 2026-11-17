"""Read recordings: rsrec episode files (schema rsrec/1.2; 1.1 also readable) via the vendored gate7 reader.

Every episode becomes the same dict: {path, header, steps: [{t, state, contacts?, vel?, rand?, src?}], footer, plans, source};
the detectors only know this one shape. Build such dicts yourself to feed other recordings to `check_episode` / `check_batch`.
(The official-evidence jsonl reader of the original tool depends on internal tooling and is not part of this package.)
"""
from __future__ import annotations

import glob
import os


def gate7():
    """The vendored gate7 module (R0-R8 rules and the rsrec reader)."""
    from . import gate7 as G
    return G


# ------------------------------------------------------------------ rsrec
def rsrec_files(root):
    """All episode files under root (recursive; includes .part files left behind by a crash)."""
    fs = sorted(glob.glob(os.path.join(root, "**", "ep_*.jsonl.gz"), recursive=True))
    fs += sorted(glob.glob(os.path.join(root, "**", "ep_*.jsonl.gz.part"), recursive=True))
    return fs


def load_rsrec(path):
    ep = gate7().load_episode(path)
    ep["source"] = "rsrec"
    ep["partial"] = str(path).endswith(".part")
    return ep
