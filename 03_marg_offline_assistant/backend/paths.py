from __future__ import annotations

import sys
from pathlib import Path


MODEL_ID = "dpp-gita-rag-assistant-v2"
INTENT_MODEL_ID = "dpp-marg-intent-small-v1"


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        bundle_root = Path(sys.executable).resolve().parents[1] / "Resources"
        if bundle_root.exists():
            return bundle_root
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def model_dir() -> Path:
    return app_root() / "models" / MODEL_ID


def intent_model_dir() -> Path:
    return app_root() / "models" / INTENT_MODEL_ID
