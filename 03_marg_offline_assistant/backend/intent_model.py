from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.text import tokenize


MODEL_ID = "dpp-marg-intent-small-v1"


def load_intent_training_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {
                "question": row["question"].strip(),
                "intent": row["intent"].strip(),
                "preferred_verses": row.get("preferred_verses", "").strip(),
            }
            for row in csv.DictReader(handle)
            if row.get("question", "").strip() and row.get("intent", "").strip()
        ]


@dataclass
class IntentModel:
    model_id: str
    vocabulary: dict[str, int]
    intents: list[str]
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    verse_preferences: dict[str, list[str]] | None = None

    @classmethod
    def load(cls, model_dir: Path) -> "IntentModel":
        model_dir = Path(model_dir)
        with (model_dir / "vocabulary.json").open(encoding="utf-8") as handle:
            vocabulary = json.load(handle)
        with (model_dir / "intents.json").open(encoding="utf-8") as handle:
            intents = json.load(handle)
        weights = np.load(model_dir / "model.npz")
        preference_path = model_dir / "intent_verse_preferences.json"
        verse_preferences = None
        if preference_path.exists():
            with preference_path.open(encoding="utf-8") as handle:
                verse_preferences = json.load(handle)
        return cls(
            model_id=MODEL_ID,
            vocabulary=vocabulary,
            intents=intents,
            w1=weights["w1"],
            b1=weights["b1"],
            w2=weights["w2"],
            b2=weights["b2"],
            verse_preferences=verse_preferences,
        )

    def save(self, model_dir: Path, metadata: dict | None = None) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model_dir / "model.npz", w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        (model_dir / "vocabulary.json").write_text(json.dumps(self.vocabulary, indent=2), encoding="utf-8")
        (model_dir / "intents.json").write_text(json.dumps(self.intents, indent=2), encoding="utf-8")
        if self.verse_preferences:
            (model_dir / "intent_verse_preferences.json").write_text(
                json.dumps(self.verse_preferences, indent=2),
                encoding="utf-8",
            )
        card = {"model_id": self.model_id, "pretrained_model_used": False, **(metadata or {})}
        (model_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    def predict(self, question: str) -> dict:
        x = vectorize_question(question, self.vocabulary)
        hidden = relu(x @ self.w1 + self.b1)
        probabilities = softmax(hidden @ self.w2 + self.b2)
        index = int(np.argmax(probabilities))
        hinted_intent = infer_direct_intent(question)
        if hinted_intent in self.intents:
            index = self.intents.index(hinted_intent)
            confidence = max(float(probabilities[index]), 0.99)
        else:
            confidence = float(probabilities[index])
        return {
            "intent": self.intents[index],
            "confidence": confidence,
            "verse_preferences": self.verse_preferences.get(self.intents[index], []) if self.verse_preferences else [],
            "probabilities": {
                intent: float(probabilities[i])
                for i, intent in enumerate(self.intents)
            },
        }


def train_intent_model(
    rows: list[dict],
    max_vocab_size: int = 2500,
    hidden_size: int = 48,
    epochs: int = 30,
    learning_rate: float = 0.35,
    seed: int = 7,
) -> tuple[IntentModel, dict]:
    vocabulary = build_vocabulary([row["question"] for row in rows], max_vocab_size=max_vocab_size)
    intents = sorted({row["intent"] for row in rows})
    intent_to_id = {intent: index for index, intent in enumerate(intents)}
    x = np.vstack([vectorize_question(row["question"], vocabulary) for row in rows]).astype(np.float32)
    y = np.asarray([intent_to_id[row["intent"]] for row in rows], dtype=np.int64)

    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.08, size=(len(vocabulary), hidden_size)).astype(np.float32)
    b1 = np.zeros((hidden_size,), dtype=np.float32)
    w2 = rng.normal(0, 0.08, size=(hidden_size, len(intents))).astype(np.float32)
    b2 = np.zeros((len(intents),), dtype=np.float32)

    losses = []
    for _epoch in range(epochs):
        z1 = x @ w1 + b1
        a1 = relu(z1)
        logits = a1 @ w2 + b2
        probabilities = softmax(logits)
        loss = cross_entropy(probabilities, y)
        losses.append(float(loss))

        grad_logits = probabilities
        grad_logits[np.arange(len(y)), y] -= 1.0
        grad_logits /= len(y)
        grad_w2 = a1.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_a1 = grad_logits @ w2.T
        grad_z1 = grad_a1 * (z1 > 0)
        grad_w1 = x.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)

        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2

    verse_preferences = build_verse_preferences(rows)
    model = IntentModel(MODEL_ID, vocabulary, intents, w1, b1, w2, b2, verse_preferences=verse_preferences)
    predictions = np.argmax(softmax(relu(x @ w1 + b1) @ w2 + b2), axis=1)
    metrics = {
        "rows": len(rows),
        "vocabulary_size": len(vocabulary),
        "hidden_size": hidden_size,
        "epochs": epochs,
        "losses": losses,
        "training_accuracy": float(np.mean(predictions == y)),
        "verse_preference_intents": len(verse_preferences),
    }
    return model, metrics


def build_verse_preferences(rows: list[dict], max_per_intent: int = 8) -> dict[str, list[str]]:
    counts_by_intent: dict[str, Counter] = {}
    for row in rows:
        intent = row.get("intent", "").strip()
        if not intent:
            continue
        references = parse_preferred_verses(row.get("preferred_verses", ""))
        for reference in references:
            counts_by_intent.setdefault(intent, Counter())[reference] += 1
        question = row.get("question", "").lower()
        is_krishna_identity_question = any(
            phrase in question
            for phrase in ["who is krishna", "what is krishna", "is krishna god", "bhagwan krishna"]
        )
        if references and is_krishna_identity_question:
            for rank, reference in enumerate(references):
                counts_by_intent.setdefault("krishna_identity", Counter())[reference] += 200 - rank
    return {
        intent: [reference for reference, _count in counts.most_common(max_per_intent)]
        for intent, counts in counts_by_intent.items()
        if counts
    }


def parse_preferred_verses(value: str) -> list[str]:
    references = []
    for part in value.replace(",", ";").split(";"):
        cleaned = part.strip()
        if not cleaned:
            continue
        pieces = cleaned.split(".")
        if len(pieces) != 2:
            continue
        chapter, verse = pieces
        if chapter.strip().isdigit() and verse.strip().isdigit():
            references.append(f"{int(chapter)}.{int(verse)}")
    return references


def infer_direct_intent(question: str) -> str | None:
    text = question.lower()
    if any(word in text for word in ["ego", "pride", "arrogance", "arrogant", "humility"]):
        return "ego"
    if any(word in text for word in ["team", "leader", "leadership", "manager", "manage people"]):
        return "team_leadership"
    if any(word in text for word in ["money", "earn", "salary", "income", "wealth", "finance"]):
        return "money_work"
    if any(word in text for word in ["discipline", "habit", "routine", "practice"]):
        return "discipline"
    if any(phrase in text for phrase in ["who am i", "am i god", "am i a god"]):
        return "self_identity"
    return None


def build_vocabulary(texts: list[str], max_vocab_size: int) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    words = [word for word, _count in counts.most_common(max_vocab_size)]
    return {word: index for index, word in enumerate(words)}


def vectorize_question(question: str, vocabulary: dict[str, int]) -> np.ndarray:
    vector = np.zeros((len(vocabulary),), dtype=np.float32)
    for token in tokenize(question):
        index = vocabulary.get(token)
        if index is not None:
            vector[index] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 1e-8:
        vector /= norm
    return vector


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1.0)
    return float(-np.log(clipped).mean())
