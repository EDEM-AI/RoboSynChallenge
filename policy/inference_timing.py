"""Timing utilities shared by policy evaluation adapters."""

from __future__ import annotations

import time

import torch


def _synchronize(device):
    if device is None or not str(device).startswith("cuda"):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _block_until_ready(value):
    block_until_ready = getattr(value, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()
    elif isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready(item)


def timed_inference(function, *args, device=None, **kwargs):
    """Run one inference boundary and return its synchronized wall-clock latency."""
    _synchronize(device)
    started_at = time.perf_counter()
    value = function(*args, **kwargs)
    _block_until_ready(value)
    _synchronize(device)
    return value, time.perf_counter() - started_at
