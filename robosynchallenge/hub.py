"""Hugging Face locations for the published challenge assets."""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id


HUB_OWNER = "han-xudong"
DATASET_REVISION = "v3.0"
DEFAULT_SIM_DATASET = f"{HUB_OWNER}/cobotmagic_Sim_click_bell"
CHECKPOINT_PREFIXES = {"act": "ACT", "dp": "DP", "smolvla": "SmolVLA"}


def is_hub_id(reference):
    if reference.count("/") != 1:
        return False
    try:
        validate_repo_id(reference)
    except HFValidationError:
        return False
    return True


def dataset_source(reference):
    """Use a local dataset, a Hub ID, or a released simulation task name."""
    path = Path(reference).expanduser()
    if (path / "meta/info.json").is_file():
        return "local/robosynchallenge", path.resolve()
    if path.exists() or reference.startswith((".", "~", "/")):
        raise FileNotFoundError(f"No LeRobot dataset metadata in {path}.")
    if is_hub_id(reference):
        return reference, None
    if reference.startswith(("cobotmagic_Sim_", "cobotmagic_Real_")):
        repo_id = f"{HUB_OWNER}/{reference}"
    else:
        repo_id = f"{HUB_OWNER}/cobotmagic_Sim_{reference}"
    validate_repo_id(repo_id)
    return repo_id, None


def checkpoint_directory(policy, task, reference=None):
    """Resolve a saved checkpoint directory or download a published model."""
    if reference is not None:
        reference = str(reference)
        path = Path(reference).expanduser()
        if path.is_dir():
            nested = path / "pretrained_model"
            return (nested if nested.is_dir() else path).resolve()
        if path.exists() or reference.startswith((".", "~", "/")) or not is_hub_id(reference):
            raise FileNotFoundError(f"Checkpoint directory is missing: {path}")
        repo_id = reference
    else:
        if "/" in task:
            raise ValueError(f"Provide a checkpoint directory or Hub model ID for {task}.")
        repo_id = f"{HUB_OWNER}/{CHECKPOINT_PREFIXES[policy]}_sim_{task}"

    print(f"Loading Hugging Face checkpoint {repo_id}", flush=True)
    return Path(snapshot_download(repo_id))
