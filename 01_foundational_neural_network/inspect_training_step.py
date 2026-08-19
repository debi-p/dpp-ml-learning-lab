import argparse
import json

from predict import DEFAULT_MODEL_ID
from sdk.classifier import EmailClassifier


def main():
    parser = argparse.ArgumentParser(description="Inspect one Phase 1 model training step.")
    parser.add_argument("message", help="Message text to inspect.")
    parser.add_argument("correct_label", choices=["work", "personal", "promotion", "spam"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Saved model ID to use.")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    classifier = EmailClassifier.load(args.model_id)
    trace = classifier.inspect_training_step(
        args.message,
        correct_label=args.correct_label,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
