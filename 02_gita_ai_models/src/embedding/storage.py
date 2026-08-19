import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.dataset import GitaVerse
from src.embedding.model import EmbeddingModel
from src.embedding.search import EmbeddingSearchIndex


def save_embedding_artifacts(model_dir, model_id, model, vocabulary, verses, verse_embeddings, config, metrics):
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output / "model.npz",
        token_embeddings=model.token_embeddings,
        projection=model.projection,
        pad_id=np.array([model.pad_id], dtype=np.int64),
    )
    np.save(output / "verse_embeddings.npy", verse_embeddings.astype(np.float32))
    (output / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    (output / "verse_index.json").write_text(json.dumps([asdict(verse) for verse in verses], indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "model_card.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "type": "neural_embedding_from_scratch",
                "algorithm": "token embedding + mean pooling + dense projection + triplet loss",
                "pretrained_model_used": False,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_embedding_artifacts(model_dir):
    root = Path(model_dir)
    card = json.loads((root / "model_card.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    verse_rows = json.loads((root / "verse_index.json").read_text(encoding="utf-8"))
    weights = np.load(root / "model.npz")
    model = EmbeddingModel(
        token_embeddings=weights["token_embeddings"].astype(np.float32),
        projection=weights["projection"].astype(np.float32),
        pad_id=int(weights["pad_id"][0]),
    )
    verse_embeddings = np.load(root / "verse_embeddings.npy").astype(np.float32)
    verses = [GitaVerse(**row) for row in verse_rows]
    return EmbeddingSearchIndex(
        model_id=card["model_id"],
        model=model,
        vocabulary=vocabulary,
        verses=verses,
        verse_embeddings=verse_embeddings,
        max_length=int(config.get("max_length", 32)),
    )
