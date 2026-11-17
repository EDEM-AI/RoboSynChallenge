"""env_anomaly: per-bug detectors for known environment bugs of the official simulation (rule ids E01-E32, local to this tool), separating environment errors from executor errors, plus a training-view export."""
VERSION = "0.1"
from .run import check_episode, check_batch, train_view  # noqa: E402,F401
