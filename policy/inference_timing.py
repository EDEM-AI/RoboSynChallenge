"""Synchronized timing shared by policy adapters."""

import time

import torch


def start_inference(device=None):
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def finish_inference(started_at, samples, device=None):
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    samples.append(time.perf_counter() - started_at)
