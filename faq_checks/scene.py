"""A small, simulator-independent view of one env step.

``FlatScene`` wraps a flat dict of floats plus optional contact rows:

    obj_<uid>_{x,y,z,qw,qx,qy,qz}              rigid object pose (world, metres, wxyz)
    art_<uid>_{x,y,z,qw,qx,qy,qz}              articulation root pose
    art_<uid>_qpos_<j>                          articulation joint position j
    artlink_<uid>_<link>_{x,y,z,qw,qx,qy,qz}   articulation link pose
    link_<left|right>_eef_{x,y,z,qw,qx,qy,qz}   robot wrist flange (link6) pose

    contacts: None (unknown) or a list of rows [name_a, name_b, n_points, ...], one per unordered pair
              of touching bodies.  Names are ``uid``, ``uid:link`` or ``robot:link``.
    events:   set of event names that happened on this step (``"weld"`` for item_assembly).

``sim_reader.py`` builds these from a live EmbodiChain env; offline tools can build them from stored states.
Task modules only talk to the helpers below, so they never touch the simulator themselves.
"""
import numpy as np

from . import common as C

POSE_AXES = ("x", "y", "z", "qw", "qx", "qy", "qz")


class FlatScene:
    def __init__(self, state, contacts=None, events=(), t=0):
        self.state = state
        self.contacts = contacts
        self.events = set(events)
        self.t = int(t)

    # ---- poses
    def has(self, key):
        return key in self.state

    def _pose(self, prefix):
        s = self.state
        return C.T_from_pos_quat([s[prefix + "x"], s[prefix + "y"], s[prefix + "z"]],
                                 [s[prefix + "qw"], s[prefix + "qx"], s[prefix + "qy"], s[prefix + "qz"]])

    def obj_T(self, uid):
        return self._pose("obj_" + uid + "_")

    def art_T(self, uid):
        return self._pose("art_" + uid + "_")

    def art_link_T(self, uid, link):
        return self._pose("artlink_" + uid + "_" + link + "_")

    def art_pos(self, uid):
        s = self.state
        return np.array([s["art_" + uid + "_" + a] for a in "xyz"], float)

    def art_qpos(self, uid, j=0, default=None):
        return self.state.get("art_%s_qpos_%d" % (uid, j), default)

    def eef_T(self, side):
        return self._pose("link_" + side + "_eef_")

    def eef_pos(self, side):
        return self.eef_T(side)[:3, 3]

    # ---- contacts
    def robot_touching(self, uid):
        """Does any robot link touch ``uid`` on this step?  None if contacts are not available."""
        if self.contacts is None:
            return None
        for r in self.contacts:
            a, b = str(r[0]), str(r[1])
            if (a == uid and b.startswith("robot:")) or (b == uid and a.startswith("robot:")):
                return True
        return False

    def touching(self, a_uid, b_uid):
        if self.contacts is None:
            return None
        return any({str(r[0]), str(r[1])} == {a_uid, b_uid} for r in self.contacts)

    def pair_set(self):
        """All contact pairs as a set of (a, b) names, both orders."""
        ps = set()
        for c in self.contacts or []:
            ps.add((str(c[0]), str(c[1])))
            ps.add((str(c[1]), str(c[0])))
        return ps
