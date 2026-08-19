import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PackagedModel:
    model_id: str
    vocabulary: dict
    verses: list
    verse_embeddings: np.ndarray
    token_embeddings: np.ndarray
    projection: np.ndarray
    max_length: int


def load_packaged_model(model_dir):
    root = Path(model_dir)
    card = json.loads((root / "model_card.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    verses = json.loads((root / "verse_index.json").read_text(encoding="utf-8"))
    verse_embeddings = np.load(root / "verse_embeddings.npy").astype(np.float32)
    weights = np.load(root / "model.npz")
    return PackagedModel(
        model_id="dpp-gita-rag-assistant-v2",
        vocabulary=vocabulary,
        verses=verses,
        verse_embeddings=verse_embeddings,
        token_embeddings=weights["token_embeddings"].astype(np.float32),
        projection=weights["projection"].astype(np.float32),
        max_length=int(config.get("max_length", 48)),
    )
