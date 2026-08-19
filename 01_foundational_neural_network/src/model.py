import numpy as np


def relu(values):
    return np.maximum(0.0, values)


def relu_derivative(values):
    return (values > 0.0).astype(float)


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def one_hot(labels, num_classes):
    encoded = np.zeros((len(labels), num_classes), dtype=float)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def cross_entropy_loss(probabilities, labels, class_weights=None):
    epsilon = 1e-12
    clipped = np.clip(probabilities, epsilon, 1.0)
    correct_class_probabilities = clipped[np.arange(len(labels)), labels]
    losses = -np.log(correct_class_probabilities)
    if class_weights is not None:
        sample_weights = class_weights[labels]
        return float(np.sum(losses * sample_weights) / np.sum(sample_weights))
    return float(np.mean(losses))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, hidden_size))
        self.b1 = np.zeros(hidden_size, dtype=float)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_size), size=(hidden_size, output_size))
        self.b2 = np.zeros(output_size, dtype=float)

    def forward(self, inputs):
        self.Z1 = inputs @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def compute_gradients(self, inputs, labels, class_weights=None):
        sample_count = inputs.shape[0]
        probabilities = self.forward(inputs)
        loss = cross_entropy_loss(probabilities, labels, class_weights=class_weights)

        target = one_hot(labels, self.W2.shape[1])
        if class_weights is None:
            dZ2 = (probabilities - target) / sample_count
        else:
            sample_weights = class_weights[labels].reshape(-1, 1)
            dZ2 = (probabilities - target) * sample_weights / np.sum(sample_weights)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = inputs.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {
            "loss": loss,
            "probabilities": probabilities,
            "dZ2": dZ2,
            "dW2": dW2,
            "db2": db2,
            "dA1": dA1,
            "dZ1": dZ1,
            "dW1": dW1,
            "db1": db1,
        }

    def train_step(self, inputs, labels, learning_rate, class_weights=None):
        gradients = self.compute_gradients(inputs, labels, class_weights=class_weights)

        self.W2 -= learning_rate * gradients["dW2"]
        self.b2 -= learning_rate * gradients["db2"]
        self.W1 -= learning_rate * gradients["dW1"]
        self.b1 -= learning_rate * gradients["db1"]

        return gradients["loss"]

    def predict_probabilities(self, inputs):
        return self.forward(inputs)

    def predict_classes(self, inputs):
        probabilities = self.predict_probabilities(inputs)
        return np.argmax(probabilities, axis=1)
