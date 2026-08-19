import argparse
import json

from predict import DEFAULT_MODEL_ID
from sdk.classifier import EmailClassifier


def main():
    parser = argparse.ArgumentParser(description="Inspect a Phase 1 model forward pass.")
    parser.add_argument("message", help="Message text to inspect.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Saved model ID to use.")
    args = parser.parse_args()

    classifier = EmailClassifier.load(args.model_id)
    trace = classifier.inspect_forward(args.message)
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
