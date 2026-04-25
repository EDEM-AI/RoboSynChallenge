#!/usr/bin/env python3
"""Convert camera transforms to EmbodiChain camera extrinsics (eye/target/up).

Usage style:
- Edit the matrix variables in the USER SETTINGS section below.
- Run this script directly.
- The script prints eye/target/up and a ready-to-paste extrinsics snippet.

Convention:
- T_a_b maps points from frame b into frame a (p_a = T_a_b * p_b).
"""

from __future__ import annotations

import json
from typing import Iterable, List

import numpy as np


# ===================== USER SETTINGS =====================
# Camera transform in arm-base frame: T_arm_cam
T_ARM_CAM = np.array(
    [
    [-0.150611, -0.917003, 0.369354, -0.026069],
    [-0.988588, 0.140871, -0.053373, -0.247272],
    [-0.003088, -0.373178, -0.927755, 0.641134],
    [0.000000, 0.000000, 0.000000, 1.000000]
    ],
    dtype=np.float64,
)

# Arm-base transform in world frame: T_world_arm
T_WORLD_ARM = np.array(
    [
        [1.0, 0.0, 0.0, 0.233],
        [0.0, 1.0, 0.0, 0.300],
        [0.0, 0.0, 1.0, 0.76],
        [0.0, 0.0, 0.0, 1.000],
    ],
    dtype=np.float64,
)

# If your provided matrix is inverse form, toggle these flags.
CAM_IN_ARM_IS_INVERSE = False   # True means current T_ARM_CAM is T_cam_arm
ARM_IN_WORLD_IS_INVERSE = False # True means current T_WORLD_ARM is T_arm_world

# Camera local axes for look-at construction.
FORWARD_AXIS = "+z"
UP_AXIS = "+y"

# target = eye + TARGET_DISTANCE * forward_world
TARGET_DISTANCE = 1.0

# Print precision
DIGITS = 6
# =========================================================


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Zero-length vector cannot be normalized: {v}")
    return v / n


def _axis_from_name(name: str) -> np.ndarray:
    table = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    key = name.strip().lower()
    if key not in table:
        raise ValueError(f"Unsupported axis '{name}'. Use one of: {', '.join(table.keys())}")
    return table[key]


def compute_eye_target_up(
    t_arm_cam: np.ndarray,
    t_world_arm: np.ndarray,
    forward_axis_local: np.ndarray,
    up_axis_local: np.ndarray,
    target_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute eye/target/up from transform chain."""
    t_world_cam = t_world_arm @ t_arm_cam

    r_world_cam = t_world_cam[:3, :3]
    eye = t_world_cam[:3, 3]

    forward_world = _normalize(r_world_cam @ forward_axis_local)
    up_world = _normalize(r_world_cam @ up_axis_local)

    # Ensure up is orthogonalized to forward for a stable look_at definition.
    up_world = _normalize(up_world - np.dot(up_world, forward_world) * forward_world)

    target = eye + target_distance * forward_world
    return eye, target, up_world, t_world_cam


def _format_vec(v: Iterable[float], digits: int) -> List[float]:
    return [round(float(x), digits) for x in v]


def main() -> None:
    t_arm_cam = np.array(T_ARM_CAM, dtype=np.float64)
    t_world_arm = np.array(T_WORLD_ARM, dtype=np.float64)

    if t_arm_cam.shape != (4, 4):
        raise ValueError(f"T_ARM_CAM must be 4x4, got {t_arm_cam.shape}")
    if t_world_arm.shape != (4, 4):
        raise ValueError(f"T_WORLD_ARM must be 4x4, got {t_world_arm.shape}")

    if CAM_IN_ARM_IS_INVERSE:
        t_arm_cam = np.linalg.inv(t_arm_cam)
    if ARM_IN_WORLD_IS_INVERSE:
        t_world_arm = np.linalg.inv(t_world_arm)

    forward_axis_local = _axis_from_name(FORWARD_AXIS)
    up_axis_local = _axis_from_name(UP_AXIS)

    if abs(np.dot(_normalize(forward_axis_local), _normalize(up_axis_local))) > 1e-6:
        raise ValueError("forward-axis and up-axis must be orthogonal.")

    eye, target, up, t_world_cam = compute_eye_target_up(
        t_arm_cam=t_arm_cam,
        t_world_arm=t_world_arm,
        forward_axis_local=forward_axis_local,
        up_axis_local=up_axis_local,
        target_distance=TARGET_DISTANCE,
    )

    eye_l = _format_vec(eye, DIGITS)
    target_l = _format_vec(target, DIGITS)
    up_l = _format_vec(up, DIGITS)

    print("# Computed camera extrinsics")
    print(json.dumps({"eye": eye_l, "target": target_l, "up": up_l}, ensure_ascii=False, indent=2))

    print("\n# Ready-to-paste config snippet")
    snippet = {"extrinsics": {"eye": eye_l, "target": target_l, "up": up_l}}
    print(json.dumps(snippet, ensure_ascii=False, indent=2))

    print("\n# Derived T_world_cam")
    print(np.array2string(t_world_cam, precision=DIGITS, suppress_small=False))


if __name__ == "__main__":
    main()
