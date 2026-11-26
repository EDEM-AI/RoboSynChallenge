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


def deployment_from_path(path):
    """Translate an in-repository legacy scene path to its native deployment."""
    path = Path(path).expanduser().resolve()
    if path.suffix != ".json" or not path.is_relative_to(ROOT / "configs"):
        return path
    directory = path.parent.relative_to(ROOT / "configs")
    parts = list(directory.parts)
    if not parts:
        return path
    setting = parts.pop() if parts[-1] in {"clear", "random"} else None
    if setting is None:
        setting = "random" if "random" in path.stem else "clear"
    return deployment_path("/".join(parts), setting)
