import argparse
from pathlib import Path

from backend.intent_model import load_intent_training_rows, train_intent_model


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train dpp-marg-intent-small-v1 from scratch.")
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[
            ROOT / "data" / "marg_intent_questions.csv",
            ROOT / "data" / "marg_intent_questions_hard_10k.csv",
            ROOT / "data" / "marg_intent_verses_20k.csv",
        ],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "dpp-marg-intent-small-v1")
    parser.add_argument("--max-vocab-size", type=int, default=2500)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    args = parser.parse_args()

    rows = []
    for data_path in args.data:
        rows.extend(load_intent_training_rows(data_path))
    model, metrics = train_intent_model(
        rows,
        max_vocab_size=args.max_vocab_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    metrics["data_files"] = [str(path) for path in args.data]
    model.save(args.output, metadata=metrics)
    print(metrics)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
