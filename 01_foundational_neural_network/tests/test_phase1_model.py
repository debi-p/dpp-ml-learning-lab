import unittest

import numpy as np

from src.model import NeuralNetwork, cross_entropy_loss, one_hot, relu, softmax


class Phase1ModelTests(unittest.TestCase):
    def test_relu_sets_negative_values_to_zero(self):
        values = np.array([[-2.0, 0.0, 3.0]])

        result = relu(values)

        np.testing.assert_array_equal(result, np.array([[0.0, 0.0, 3.0]]))

    def test_softmax_returns_probabilities_per_row(self):
        logits = np.array([[1.0, 2.0, 3.0]])

        probabilities = softmax(logits)

        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertGreater(probabilities[0, 2], probabilities[0, 1])

    def test_one_hot_encodes_class_indices(self):
        encoded = one_hot(np.array([0, 2]), num_classes=3)

        np.testing.assert_array_equal(encoded, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

    def test_cross_entropy_loss_is_lower_for_better_predictions(self):
        labels = np.array([1])
        weak_prediction = np.array([[0.33, 0.34, 0.33]])
        strong_prediction = np.array([[0.05, 0.90, 0.05]])

        weak_loss = cross_entropy_loss(weak_prediction, labels)
        strong_loss = cross_entropy_loss(strong_prediction, labels)

        self.assertLess(strong_loss, weak_loss)

    def test_neural_network_forward_returns_class_probabilities(self):
        model = NeuralNetwork(input_size=4, hidden_size=3, output_size=2, seed=7)
        inputs = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

        probabilities = model.forward(inputs)

        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)

    def test_training_step_updates_weights_and_returns_loss(self):
        model = NeuralNetwork(input_size=4, hidden_size=3, output_size=2, seed=7)
        inputs = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        labels = np.array([0, 1])
        old_w1 = model.W1.copy()

        loss = model.train_step(inputs, labels, learning_rate=0.1)

        self.assertGreater(loss, 0.0)
        self.assertFalse(np.array_equal(old_w1, model.W1))

    def test_training_step_accepts_class_weights_for_imbalanced_data(self):
        model = NeuralNetwork(input_size=4, hidden_size=3, output_size=2, seed=7)
        inputs = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        labels = np.array([0, 1])
        class_weights = np.array([0.5, 2.0])

        loss = model.train_step(inputs, labels, learning_rate=0.1, class_weights=class_weights)

        self.assertGreater(loss, 0.0)

    def test_compute_gradients_returns_backpropagation_details(self):
        model = NeuralNetwork(input_size=4, hidden_size=3, output_size=2, seed=7)
        inputs = np.array([[1.0, 0.0, 1.0, 0.0]])
        labels = np.array([1])

        details = model.compute_gradients(inputs, labels)

        self.assertGreater(details["loss"], 0.0)
        self.assertEqual(details["dZ2"].shape, (1, 2))
        self.assertEqual(details["dW2"].shape, (3, 2))
        self.assertEqual(details["db2"].shape, (2,))
        self.assertEqual(details["dA1"].shape, (1, 3))
        self.assertEqual(details["dZ1"].shape, (1, 3))
        self.assertEqual(details["dW1"].shape, (4, 3))
        self.assertEqual(details["db1"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
