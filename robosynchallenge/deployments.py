"""Locate native RoboSynChallenge deployments by task and setting."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = re.compile(r"(?:other_tasks/)?[a-z][a-z0-9_]*")


def deployment_path(task, setting):
    if TASK_NAME.fullmatch(task) is None or setting not in {"clear", "random"}:
        raise ValueError("Expected a task identifier and clear or random.")
    for group in ("", "other_tasks"):
        path = ROOT / "configs" / group / task / setting / "task.cobotmagic.yaml"
        if path.is_file():
            return path
        if "/" in task:
            break
    raise FileNotFoundError(f"No native deployment for {task}/{setting}.")


def require_deployment_path(path):
    """Resolve a native task deployment supplied on the command line."""
    path = Path(path).expanduser().resolve()
    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a task deployment in YAML: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Task deployment does not exist: {path}")
    return path
