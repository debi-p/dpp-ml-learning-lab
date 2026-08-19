import argparse

from sdk.classifier import EmailClassifier


DEFAULT_MODEL_ID = "dpp-email-classifier-small-v1"


def predict_text(model_id, message):
    classifier = EmailClassifier.load(model_id)
    return classifier.predict(message)


def main():
    parser = argparse.ArgumentParser(description="Predict message class with a saved Phase 1 model.")
    parser.add_argument("message", help="Message text to classify.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Saved model ID to use.")
    args = parser.parse_args()

    result = predict_text(args.model_id, args.message)
    print(f"Model: {result['model_id']}")
    print(f"Prediction: {result['prediction']}")
    print("\nConfidence:")
    for label, score in result["confidence"].items():
        print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
