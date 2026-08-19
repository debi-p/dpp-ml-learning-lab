import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.dataset import load_verses_csv
from src.embedding.model import EmbeddingModel
from src.embedding.storage import save_embedding_artifacts
from src.embedding.training_data import build_training_examples, load_training_pairs, validate_pairs_against_verses
from src.embedding.vocabulary import build_embedding_vocabulary, encode_text


MODEL_ID = "dpp-gita-embedding-small-v1"


def verse_text(verse):
    return " ".join([verse.translation, verse.commentary, verse.tags]).strip()


def build_text_corpus(pairs, verses):
    texts = []
    texts.extend(pair.question for pair in pairs)
    texts.extend(pair.answer for pair in pairs)
    texts.extend(pair.topic for pair in pairs)
    texts.extend(verse_text(verse) for verse in verses)
    return texts


def precompute_verse_embeddings(model, vocabulary, verses, max_length):
    return np.vstack([model.embed_text(verse_text(verse), vocabulary, max_length=max_length) for verse in verses])


def train_embedding_model(
    pairs_path="data/gita_question_pairs.csv",
    verses_path="data/gita_verses.csv",
    model_dir=f"models/{MODEL_ID}",
    max_vocab_size=8000,
    max_length=48,
    token_dim=32,
    output_dim=64,
    epochs=6,
    learning_rate=0.03,
    margin=0.25,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    verses = load_verses_csv(verses_path)
    pairs = load_training_pairs(pairs_path)
    matched, unmatched = validate_pairs_against_verses(pairs, verses)
    if unmatched:
        raise SystemExit(f"{len(unmatched)} training rows do not match a parsed verse.")

    vocabulary = build_embedding_vocabulary(build_text_corpus(matched, verses), max_size=max_vocab_size)
    examples = build_training_examples(matched, verses, seed=seed)
    encoded = [
        (
            encode_text(example.question, vocabulary, max_length=max_length),
            encode_text(verse_text(example.positive_verse), vocabulary, max_length=max_length),
            encode_text(verse_text(example.negative_verse), vocabulary, max_length=max_length),
        )
        for example in examples
    ]

    model = EmbeddingModel.create(
        vocab_size=len(vocabulary),
        token_dim=token_dim,
        output_dim=output_dim,
        seed=seed,
    )

    losses = []
    for epoch in range(1, epochs + 1):
        random.shuffle(encoded)
        epoch_loss = 0.0
        active = 0
        for question_ids, positive_ids, negative_ids in encoded:
            loss = model.train_triplet_step(
                question_ids,
                positive_ids,
                negative_ids,
                learning_rate=learning_rate,
                margin=margin,
            )
            epoch_loss += loss
            if loss > 0:
                active += 1
        average_loss = epoch_loss / max(1, len(encoded))
        losses.append(average_loss)
        print(f"epoch={epoch} loss={average_loss:.6f} active_triplets={active}")

    verse_embeddings = precompute_verse_embeddings(model, vocabulary, verses, max_length=max_length)
    config = {
        "model_id": MODEL_ID,
        "max_length": max_length,
        "max_vocab_size": max_vocab_size,
        "token_dim": token_dim,
        "output_dim": output_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "margin": margin,
        "seed": seed,
    }
    metrics = {
        "training_rows": len(matched),
        "verse_rows": len(verses),
        "vocabulary_size": len(vocabulary),
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
    }
    save_embedding_artifacts(
        model_dir=Path(model_dir),
        model_id=MODEL_ID,
        model=model,
        vocabulary=vocabulary,
        verses=verses,
        verse_embeddings=verse_embeddings,
        config=config,
        metrics=metrics,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train dpp-gita-embedding-small-v1 from scratch.")
    parser.add_argument("--pairs", default="data/gita_question_pairs.csv")
    parser.add_argument("--verses", default="data/gita_verses.csv")
    parser.add_argument("--model-dir", default=f"models/{MODEL_ID}")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-vocab-size", type=int, default=8000)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=64)
    args = parser.parse_args()

    metrics = train_embedding_model(
        pairs_path=args.pairs,
        verses_path=args.verses,
        model_dir=args.model_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_vocab_size=args.max_vocab_size,
        max_length=args.max_length,
        token_dim=args.token_dim,
        output_dim=args.output_dim,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

