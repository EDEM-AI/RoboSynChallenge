"""Deploy the original OpenPI π₀.₅ model."""

from policy.pi0.deploy_policy import eval, load_model, reset_model
from .pi_model import PI0

__all__ = ["eval", "get_model", "reset_model"]


def get_model(config):
    return load_model(config, PI0)
