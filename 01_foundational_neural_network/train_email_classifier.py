from pathlib import Path

import numpy as np

from src.data import load_messages
from src.evaluate import accuracy_score, confusion_matrix
from src.features import build_vocabulary, vectorize_tokens
from src.labels import build_label
from src.model import NeuralNetwork
from src.storage import save_artifacts
from src.text import clean_text, tokenize
from src.train import compute_class_weights, train_model, train_test_split


MODEL_ID = "dpp-email-classifier-small-v1"
LABELS = ["work", "personal", "promotion", "spam"]
VOCAB_SIZE = 1000
HIDDEN_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.2


def prepare_dataset(csv_path):
    rows = load_messages(csv_path)
    labeled_rows = []

    for row in rows:
        cleaned = clean_text(row["message"])
        tokens = tokenize(cleaned)
        label = build_label(row["category"], cleaned)
        labeled_rows.append(
            {
                "message": row["message"],
                "tokens": tokens,
                "label": label,
            }
        )

    return labeled_rows


def encode_labels(labels):
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    return np.array([label_to_index[label] for label in labels], dtype=int)


def print_confusion_matrix(matrix):
    print("\nConfusion matrix")
    print("actual \\ predicted")
    print("".ljust(12) + "".join(label[:9].rjust(11) for label in LABELS))
    for index, label in enumerate(LABELS):
        row = "".join(str(value).rjust(11) for value in matrix[index])
        print(label[:9].ljust(12) + row)


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "email.csv"
    model_dir = base_dir / "models" / MODEL_ID

    dataset = prepare_dataset(csv_path)
    all_tokens = [row["tokens"] for row in dataset]
    all_labels = [row["label"] for row in dataset]

    label_indices = encode_labels(all_labels)
    row_numbers = np.arange(len(dataset))
    train_rows, y_train, test_rows, y_test = train_test_split(
        row_numbers.reshape(-1, 1), label_indices, test_ratio=0.1, seed=42
    )

    train_indices = train_rows.reshape(-1)
    test_indices = test_rows.reshape(-1)
    train_tokens = [all_tokens[index] for index in train_indices]
    test_tokens = [all_tokens[index] for index in test_indices]

    vocabulary = build_vocabulary(train_tokens, max_size=VOCAB_SIZE)
    X_train = np.vstack([vectorize_tokens(tokens, vocabulary) for tokens in train_tokens])
    X_test = np.vstack([vectorize_tokens(tokens, vocabulary) for tokens in test_tokens])

    model = NeuralNetwork(
        input_size=len(vocabulary),
        hidden_size=HIDDEN_SIZE,
        output_size=len(LABELS),
        seed=42,
    )

    print(f"Model ID: {MODEL_ID}")
    print(f"Total messages: {len(dataset)}")
    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Hidden neurons: {HIDDEN_SIZE}")
    print(f"Output classes: {', '.join(LABELS)}\n")

    class_weights = compute_class_weights(y_train, num_classes=len(LABELS))
    print(
        "Class weights: "
        + ", ".join(f"{label}={class_weights[index]:.2f}" for index, label in enumerate(LABELS))
        + "\n"
    )

    train_model(
        model,
        X_train,
        y_train,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        log_every=10,
        class_weights=class_weights,
    )

    predictions = model.predict_classes(X_test)
    accuracy = accuracy_score(predictions, y_test)
    matrix = confusion_matrix(predictions, y_test, num_classes=len(LABELS))

    print(f"\nTest accuracy: {accuracy:.4f}")
    print_confusion_matrix(matrix)

    model_card = {
        "model_id": MODEL_ID,
        "author": "Debi Prasad Pradhan",
        "phase": 1,
        "task": "message_classification",
        "classes": LABELS,
        "tokenization": "word",
        "vectorization": "bag_of_words",
        "vocab_size": len(vocabulary),
        "hidden_size": HIDDEN_SIZE,
        "output_size": len(LABELS),
        "class_weights": {
            label: float(class_weights[index]) for index, label in enumerate(LABELS)
        },
        "version": "v1",
    }
    save_artifacts(model_dir, model, vocabulary, LABELS, model_card)
    print(f"\nSaved model artifacts to: {model_dir}")


if __name__ == "__main__":
    main()
