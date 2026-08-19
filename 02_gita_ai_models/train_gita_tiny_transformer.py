import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from src.dataset import load_verses_csv
from src.text import normalize_for_display
from src.transformer.dataset import build_next_token_examples
from src.transformer.model import TinyTransformerModel
from src.transformer.storage import save_transformer_artifacts
from src.transformer.tokenizer import build_transformer_vocabulary
from src.transformer.train import train_step


MODEL_ID = "dpp-gita-tiny-transformer-v1"


def load_qa_texts(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = normalize_for_display(row.get("question") or "")
            answer = normalize_for_display(row.get("answer") or "")
            topic = normalize_for_display(row.get("topic") or "")
            text = " ".join(part for part in [question, answer, topic] if part).strip()
            if text:
                rows.append(text)
    return rows


def load_training_texts(verses_path, pairs_path, max_texts=None):
    verses = load_verses_csv(verses_path)
    verse_texts = []
    for verse in verses:
        text = " ".join([verse.translation, verse.commentary, verse.tags]).strip()
        if text:
            verse_texts.append(normalize_for_display(text))

    qa_texts = load_qa_texts(pairs_path)
    return select_balanced_texts(verse_texts, qa_texts, max_texts=max_texts)


def select_balanced_texts(verse_texts, qa_texts, max_texts=None):
    if max_texts is None:
        return list(verse_texts) + list(qa_texts)

    verse_limit = max(1, max_texts // 2)
    qa_limit = max_texts - verse_limit
    selected = _spread_sample(verse_texts, verse_limit) + _spread_sample(qa_texts, qa_limit)

    if len(selected) < max_texts:
        remaining = list(verse_texts[verse_limit:]) + list(qa_texts[qa_limit:])
        selected.extend(remaining[: max_texts - len(selected)])

    return selected


def _spread_sample(items, limit):
    items = list(items)
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]

    indexes = np.linspace(0, len(items) - 1, num=limit, dtype=int)
    return [items[int(index)] for index in indexes]


def batch_examples(examples, batch_size):
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids = np.array([example.input_ids for example in batch], dtype=np.int64)
        target_ids = np.array([example.target_id for example in batch], dtype=np.int64)
        yield input_ids, target_ids


def train_tiny_transformer(
    verses_path="data/gita_verses.csv",
    pairs_path="data/gita_question_pairs.csv",
    model_dir=f"models/{MODEL_ID}",
    max_texts=500,
    max_vocab_size=5000,
    context_length=32,
    d_model=64,
    hidden_size=128,
    epochs=2,
    batch_size=32,
    learning_rate=0.05,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    texts = load_training_texts(verses_path, pairs_path, max_texts=max_texts)
    vocabulary = build_transformer_vocabulary(texts, max_size=max_vocab_size)
    examples = build_next_token_examples(texts, vocabulary=vocabulary, context_length=context_length)
    if not examples:
        raise SystemExit("No next-token examples were created.")

    model = TinyTransformerModel.create(
        vocab_size=len(vocabulary),
        context_length=context_length,
        d_model=d_model,
        hidden_size=hidden_size,
        seed=seed,
    )

    losses = []
    for epoch in range(1, epochs + 1):
        random.shuffle(examples)
        epoch_loss = 0.0
        batches = 0
        for input_ids, target_ids in batch_examples(examples, batch_size=batch_size):
            result = train_step(model, input_ids, target_ids, learning_rate=learning_rate)
            epoch_loss += result.loss
            batches += 1
        average_loss = epoch_loss / max(1, batches)
        losses.append(average_loss)
        print(f"epoch={epoch} loss={average_loss:.6f} batches={batches}")

    config = {
        "model_id": MODEL_ID,
        "max_texts": max_texts,
        "max_vocab_size": max_vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    metrics = {
        "training_texts": len(texts),
        "training_examples": len(examples),
        "vocabulary_size": len(vocabulary),
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
    }
    save_transformer_artifacts(
        model_dir=model_dir,
        model_id=MODEL_ID,
        model=model,
        vocabulary=vocabulary,
        config=config,
        metrics=metrics,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train dpp-gita-tiny-transformer-v1 from scratch.")
    parser.add_argument("--verses", default="data/gita_verses.csv")
    parser.add_argument("--pairs", default="data/gita_question_pairs.csv")
    parser.add_argument("--model-dir", default=f"models/{MODEL_ID}")
    parser.add_argument("--max-texts", type=int, default=500)
    parser.add_argument("--max-vocab-size", type=int, default=5000)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    metrics = train_tiny_transformer(
        verses_path=args.verses,
        pairs_path=args.pairs,
        model_dir=args.model_dir,
        max_texts=args.max_texts,
        max_vocab_size=args.max_vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
