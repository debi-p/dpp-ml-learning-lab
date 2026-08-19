import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluate import accuracy_score, confusion_matrix
from src.model import NeuralNetwork
from src.storage import load_artifacts, save_artifacts
from src.train import compute_class_weights, train_test_split


class Phase1TrainingStorageTests(unittest.TestCase):
    def test_train_test_split_uses_requested_test_ratio(self):
        inputs = np.arange(20).reshape(10, 2)
        labels = np.arange(10)

        X_train, y_train, X_test, y_test = train_test_split(inputs, labels, test_ratio=0.2, seed=1)

        self.assertEqual(X_train.shape, (8, 2))
        self.assertEqual(y_train.shape, (8,))
        self.assertEqual(X_test.shape, (2, 2))
        self.assertEqual(y_test.shape, (2,))

    def test_accuracy_score_counts_correct_predictions(self):
        predicted = np.array([0, 1, 1, 3])
        actual = np.array([0, 2, 1, 3])

        self.assertEqual(accuracy_score(predicted, actual), 0.75)

    def test_confusion_matrix_counts_actual_by_predicted(self):
        predicted = np.array([0, 1, 1, 3])
        actual = np.array([0, 2, 1, 3])

        matrix = confusion_matrix(predicted, actual, num_classes=4)

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            ),
        )

    def test_compute_class_weights_gives_rare_classes_larger_weights(self):
        labels = np.array([0, 0, 0, 1])

        weights = compute_class_weights(labels, num_classes=2)

        self.assertLess(weights[0], weights[1])

    def test_save_and_load_artifacts_round_trip_model_metadata(self):
        model = NeuralNetwork(input_size=3, hidden_size=2, output_size=4, seed=3)
        vocabulary = {"free": 0, "project": 1, "home": 2}
        labels = ["work", "personal", "promotion", "spam"]
        model_card = {"model_id": "dpp-email-classifier-small-v1"}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            save_artifacts(model_dir, model, vocabulary, labels, model_card)
            loaded_model, loaded_vocabulary, loaded_labels, loaded_card = load_artifacts(model_dir)

            self.assertTrue((model_dir / "model.npz").exists())
            self.assertEqual(json.loads((model_dir / "model_card.json").read_text()), model_card)

        np.testing.assert_array_equal(loaded_model.W1, model.W1)
        self.assertEqual(loaded_vocabulary, vocabulary)
        self.assertEqual(loaded_labels, labels)
        self.assertEqual(loaded_card, model_card)


if __name__ == "__main__":
    unittest.main()
