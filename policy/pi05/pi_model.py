"""Run OpenPI π₀.₅ through the same CobotMagic observation interface as π₀."""

from policy.pi0.pi_model import PI0 as OpenPiPolicy


class PI0(OpenPiPolicy):
    policy_name = "pi05"
