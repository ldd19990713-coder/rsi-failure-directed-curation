"""Repository root resolution for the public replication package."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def get_repo_root() -> Path:
    env = os.environ.get("RSI_ROOT")
    if env:
        return Path(env).resolve()
    return REPO_ROOT


def model_id_or_path(default_hf_id: str) -> str:
    """Prefer HF hub id; allow override via RSI_MODEL_PATH / local models/ dir."""
    override = os.environ.get("RSI_MODEL_PATH")
    if override:
        return override
    local = get_repo_root() / "models" / default_hf_id.split("/")[-1]
    if local.exists():
        return str(local)
    return default_hf_id
