import json
from pathlib import Path

import numpy as np

from src.model import NeuralNetwork


def save_artifacts(model_dir, model, vocabulary, labels, model_card):
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)

    np.savez(path / "model.npz", W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    (path / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    (path / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (path / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")


def load_artifacts(model_dir):
    path = Path(model_dir)
    weights = np.load(path / "model.npz")
    vocabulary = json.loads((path / "vocabulary.json").read_text(encoding="utf-8"))
    labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))
    model_card = json.loads((path / "model_card.json").read_text(encoding="utf-8"))

    model = NeuralNetwork(
        input_size=weights["W1"].shape[0],
        hidden_size=weights["W1"].shape[1],
        output_size=weights["W2"].shape[1],
    )
    model.W1 = weights["W1"]
    model.b1 = weights["b1"]
    model.W2 = weights["W2"]
    model.b2 = weights["b2"]

    return model, vocabulary, labels, model_card
