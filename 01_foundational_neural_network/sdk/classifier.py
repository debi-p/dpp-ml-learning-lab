from pathlib import Path

import numpy as np

from src.features import vectorize_tokens
from src.storage import load_artifacts
from src.text import clean_text, tokenize


class EmailClassifier:
    def __init__(self, model, vocabulary, labels, model_card):
        self.model = model
        self.vocabulary = vocabulary
        self.labels = labels
        self.model_card = model_card

    @classmethod
    def load(cls, model_id, models_dir=None):
        if models_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
            models_dir = base_dir / "models"

        model, vocabulary, labels, model_card = load_artifacts(Path(models_dir) / model_id)
        return cls(model, vocabulary, labels, model_card)

    def metadata(self):
        return {
            "model_id": self.model_card["model_id"],
            "name": self.model_card.get("name", self.model_card["model_id"]),
            "phase": self.model_card.get("phase", 1),
            "task": self.model_card.get("task", "message_classification"),
            "task_type": "classification",
            "target_type": "class_label",
            "labels": self.labels,
            "supports": ["predict", "inspect_forward", "inspect_training_step"],
        }

    def predict(self, message):
        tokens = tokenize(clean_text(message))
        vector = vectorize_tokens(tokens, self.vocabulary).reshape(1, -1)
        probabilities = self.model.predict_probabilities(vector)[0]
        prediction_index = int(np.argmax(probabilities))

        return {
            "model_id": self.model_card["model_id"],
            "prediction": self.labels[prediction_index],
            "confidence": {
                label: float(probabilities[index]) for index, label in enumerate(self.labels)
            },
        }

    def inspect_forward(self, message):
        cleaned = clean_text(message)
        tokens = tokenize(cleaned)
        vector = vectorize_tokens(tokens, self.vocabulary).reshape(1, -1)
        probabilities = self.model.predict_probabilities(vector)[0]
        prediction_index = int(np.argmax(probabilities))

        non_zero_features = {}
        index_to_word = {index: word for word, index in self.vocabulary.items()}
        for index in np.flatnonzero(vector[0]):
            non_zero_features[index_to_word[int(index)]] = float(vector[0, index])

        output_scores = {
            label: float(self.model.Z2[0, index]) for index, label in enumerate(self.labels)
        }
        softmax_probabilities = {
            label: float(probabilities[index]) for index, label in enumerate(self.labels)
        }

        trace = {
            "model_id": self.model_card["model_id"],
            "input": {
                "raw": message,
                "cleaned": cleaned,
                "tokens": tokens,
            },
            "vectorization": {
                "algorithm": "Bag-of-Words count vector",
                "vocab_size": len(self.vocabulary),
                "non_zero_features": non_zero_features,
                "shape": f"{vector.shape[0]} x {vector.shape[1]}",
            },
            "forward_propagation": {
                "hidden_layer": {
                    "algorithm": "Z1 = XW1 + b1",
                    "shape": f"{self.model.Z1.shape[0]} x {self.model.Z1.shape[1]}",
                    "values": self.model.Z1[0].tolist(),
                },
                "activation": {
                    "algorithm": "ReLU: A1 = max(0, Z1)",
                    "shape": f"{self.model.A1.shape[0]} x {self.model.A1.shape[1]}",
                    "values": self.model.A1[0].tolist(),
                },
                "output_layer": {
                    "algorithm": "Z2 = A1W2 + b2",
                    "shape": f"{self.model.Z2.shape[0]} x {self.model.Z2.shape[1]}",
                    "scores": output_scores,
                },
                "softmax": {
                    "algorithm": "convert output scores to probabilities",
                    "probabilities": softmax_probabilities,
                },
            },
            "prediction": self.labels[prediction_index],
        }
        return self._with_generic_flow(trace)

    def inspect_training_step(self, message, correct_label, learning_rate=0.1):
        if correct_label not in self.labels:
            raise ValueError(f"Unknown label: {correct_label}")

        forward_trace = self.inspect_forward(message)
        vector = vectorize_tokens(forward_trace["input"]["tokens"], self.vocabulary).reshape(1, -1)
        label_index = self.labels.index(correct_label)
        label_array = np.array([label_index])

        before = self._weight_preview()
        gradients = self.model.compute_gradients(vector, label_array)

        after = {
            "W1": (self.model.W1 - learning_rate * gradients["dW1"])[:2, :2].tolist(),
            "b1": (self.model.b1 - learning_rate * gradients["db1"])[:4].tolist(),
            "W2": (self.model.W2 - learning_rate * gradients["dW2"])[:2, :4].tolist(),
            "b2": (self.model.b2 - learning_rate * gradients["db2"])[:4].tolist(),
        }

        trace = {
            **forward_trace,
            "training": {
                "correct_label": correct_label,
                "correct_label_index": label_index,
                "loss_algorithm": "cross-entropy loss",
                "loss": float(gradients["loss"]),
                "learning_rate": learning_rate,
            },
            "backpropagation": {
                "algorithm": "chain rule gradients",
                "dZ2": self._array_summary(gradients["dZ2"]),
                "dW2": self._array_summary(gradients["dW2"]),
                "db2": self._array_summary(gradients["db2"]),
                "dA1": self._array_summary(gradients["dA1"]),
                "dZ1": self._array_summary(gradients["dZ1"]),
                "dW1": self._array_summary(gradients["dW1"]),
                "db1": self._array_summary(gradients["db1"]),
            },
            "weight_update": {
                "algorithm": "new_value = old_value - learning_rate * gradient",
                "before": before,
                "after": after,
            },
        }
        return self._with_generic_flow(trace)

    def _with_generic_flow(self, trace):
        flow = self._build_flow(trace)
        return {
            **trace,
            "model_metadata": self.metadata(),
            "summary": self._build_summary(trace),
            "flow": flow,
        }

    def _build_summary(self, trace):
        summary = {
            "prediction": trace["prediction"],
            "vector_shape": trace["vectorization"]["shape"],
            "hidden_shape": trace["forward_propagation"]["hidden_layer"]["shape"],
        }
        if "training" in trace:
            summary["loss"] = trace["training"]["loss"]
            summary["correct_label"] = trace["training"]["correct_label"]
        return summary

    def _build_flow(self, trace):
        nodes = [
            self._node(
                "input_message",
                "Input Message",
                "raw text",
                "Original text entered by the user.",
                "User input",
                "text",
                trace["input"]["raw"],
                "Receive message",
                "Pass raw message into preprocessing.",
                "Raw message",
                "text",
                trace["input"]["raw"],
            ),
            self._node(
                "word_tokenization",
                "Word Tokenization",
                f"{len(trace['input']['tokens'])} tokens",
                "Cleans the message and splits it into words.",
                "Raw message",
                "text",
                trace["input"]["raw"],
                "lowercase + regex cleaning + whitespace split",
                "Normalize text, remove punctuation noise, split on spaces.",
                "Tokens",
                f"{len(trace['input']['tokens'])}",
                trace["input"]["tokens"],
            ),
            self._node(
                "vocabulary_lookup",
                "Vocabulary Lookup",
                f"{trace['vectorization']['vocab_size']} known words",
                "Checks which tokens exist in the trained vocabulary.",
                "Tokens",
                f"{len(trace['input']['tokens'])}",
                trace["input"]["tokens"],
                "word -> index lookup",
                "Only vocabulary words can become active numeric features.",
                "Matched vocabulary words",
                str(len(trace["vectorization"]["non_zero_features"])),
                list(trace["vectorization"]["non_zero_features"].keys()),
            ),
            self._node(
                "bag_of_words_vectorization",
                "Bag-of-Words Vectorization",
                trace["vectorization"]["shape"],
                "Converts matched words into a numeric count vector.",
                "Matched vocabulary words",
                str(len(trace["vectorization"]["non_zero_features"])),
                trace["vectorization"]["non_zero_features"],
                trace["vectorization"]["algorithm"],
                "Count how many times each known word appears.",
                "Input vector X",
                trace["vectorization"]["shape"],
                trace["vectorization"]["non_zero_features"],
            ),
            self._node(
                "hidden_layer_1",
                "Hidden Layer 1",
                self._hidden_neuron_label(trace),
                "Calculates weighted sums from the input vector.",
                "Input vector X",
                trace["vectorization"]["shape"],
                trace["vectorization"]["non_zero_features"],
                trace["forward_propagation"]["hidden_layer"]["algorithm"],
                "Multiply input features by W1 and add bias b1.",
                "Hidden weighted sum Z1",
                trace["forward_propagation"]["hidden_layer"]["shape"],
                trace["forward_propagation"]["hidden_layer"]["values"][:8],
            ),
            self._node(
                "relu_activation",
                "ReLU Activation",
                trace["forward_propagation"]["activation"]["shape"],
                "Keeps positive hidden signals and turns negative signals into zero.",
                "Hidden weighted sum Z1",
                trace["forward_propagation"]["hidden_layer"]["shape"],
                trace["forward_propagation"]["hidden_layer"]["values"][:8],
                trace["forward_propagation"]["activation"]["algorithm"],
                "Apply max(0, z) to every hidden value.",
                "Activated hidden values A1",
                trace["forward_propagation"]["activation"]["shape"],
                trace["forward_propagation"]["activation"]["values"][:8],
            ),
            self._node(
                "output_layer_2",
                "Output Layer 2",
                f"{len(self.labels)} neurons",
                "Produces one score for each possible class.",
                "Activated hidden values A1",
                trace["forward_propagation"]["activation"]["shape"],
                trace["forward_propagation"]["activation"]["values"][:8],
                trace["forward_propagation"]["output_layer"]["algorithm"],
                "Multiply A1 by W2 and add b2.",
                "Class scores Z2",
                trace["forward_propagation"]["output_layer"]["shape"],
                trace["forward_propagation"]["output_layer"]["scores"],
            ),
            self._node(
                "softmax_probabilities",
                "Softmax Probabilities",
                f"{len(self.labels)} probabilities",
                "Converts class scores into probabilities.",
                "Class scores Z2",
                trace["forward_propagation"]["output_layer"]["shape"],
                trace["forward_propagation"]["output_layer"]["scores"],
                trace["forward_propagation"]["softmax"]["algorithm"],
                "Exponentiate and normalize scores so they sum to 1.",
                "Prediction probabilities",
                f"1 x {len(self.labels)}",
                trace["forward_propagation"]["softmax"]["probabilities"],
            ),
        ]

        edges = [
            self._edge("input_message", "word_tokenization", "forward", "raw text", "text", trace["input"]["raw"]),
            self._edge("word_tokenization", "vocabulary_lookup", "forward", "tokens", f"{len(trace['input']['tokens'])}", trace["input"]["tokens"][:6]),
            self._edge("vocabulary_lookup", "bag_of_words_vectorization", "forward", "matched words", str(len(trace["vectorization"]["non_zero_features"])), list(trace["vectorization"]["non_zero_features"].keys())[:6]),
            self._edge("bag_of_words_vectorization", "hidden_layer_1", "forward", "X vector", trace["vectorization"]["shape"], trace["vectorization"]["non_zero_features"]),
            self._edge("hidden_layer_1", "relu_activation", "forward", "Z1 values", trace["forward_propagation"]["hidden_layer"]["shape"], trace["forward_propagation"]["hidden_layer"]["values"][:6]),
            self._edge("relu_activation", "output_layer_2", "forward", "A1 values", trace["forward_propagation"]["activation"]["shape"], trace["forward_propagation"]["activation"]["values"][:6]),
            self._edge("output_layer_2", "softmax_probabilities", "forward", "Z2 scores", trace["forward_propagation"]["output_layer"]["shape"], trace["forward_propagation"]["output_layer"]["scores"]),
        ]

        if "training" in trace:
            nodes.extend(
                [
                    self._node(
                        "loss_calculation",
                        "Loss Calculation",
                        "cross-entropy",
                        "Compares predicted probabilities with the correct label.",
                        "Prediction + correct label",
                        f"prediction={trace['prediction']}",
                        {"prediction": trace["prediction"], "correct_label": trace["training"]["correct_label"]},
                        trace["training"]["loss_algorithm"],
                        "Higher loss means the prediction is farther from the correct label.",
                        "Loss value",
                        "scalar",
                        trace["training"]["loss"],
                    ),
                    self._node(
                        "backpropagation",
                        "Backpropagation",
                        "chain rule gradients",
                        "Calculates how weights and biases should change.",
                        "Loss value",
                        "scalar",
                        trace["training"]["loss"],
                        trace["backpropagation"]["algorithm"],
                        "Move error backward through output and hidden layers.",
                        "Gradient summaries",
                        "multiple matrices",
                        {name: value["shape"] for name, value in trace["backpropagation"].items() if name != "algorithm"},
                    ),
                    self._node(
                        "weight_update",
                        "Weight Update",
                        f"learning rate {trace['training']['learning_rate']}",
                        "Previews the gradient descent update.",
                        "Current weights + gradients",
                        "W1/b1/W2/b2",
                        trace["weight_update"]["before"],
                        trace["weight_update"]["algorithm"],
                        "Subtract learning_rate times gradient from each trainable value.",
                        "Updated weights preview",
                        "W1/b1/W2/b2",
                        trace["weight_update"]["after"],
                    ),
                    self._node(
                        "training_step_complete",
                        "Training Step Complete",
                        "next row starts from input",
                        "This training row is complete. The next row starts from Input Message using updated weights.",
                        "Updated model state",
                        "weights ready",
                        "Updated weights are stored for the next training row.",
                        "Stop after one training step",
                        "Do not loop automatically. Click Play or Next Row later to inspect another row.",
                        "Training step status",
                        "complete",
                        "Next training row will start from Input Message.",
                    ),
                ]
            )
            edges.extend(
                [
                    self._edge("softmax_probabilities", "loss_calculation", "forward", "probabilities", f"1 x {len(self.labels)}", trace["forward_propagation"]["softmax"]["probabilities"]),
                    self._edge("loss_calculation", "backpropagation", "backward", "loss", "scalar", trace["training"]["loss"]),
                    self._edge("backpropagation", "weight_update", "backward", "gradients", "multiple matrices", {name: value["shape"] for name, value in trace["backpropagation"].items() if name != "algorithm"}),
                    self._edge("weight_update", "training_step_complete", "update", "updated weights", "model state", "next training row starts from Input Message"),
                ]
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "timeline": [node["id"] for node in nodes],
        }

    def _node(
        self,
        node_id,
        name,
        subtitle,
        description,
        input_label,
        input_shape,
        input_preview,
        algorithm_label,
        algorithm_value,
        output_label,
        output_shape,
        output_preview,
    ):
        return {
            "id": node_id,
            "name": name,
            "subtitle": subtitle,
            "description": description,
            "input": {
                "label": input_label,
                "shape": input_shape,
                "preview": input_preview,
            },
            "algorithm": {
                "label": algorithm_label,
                "value": algorithm_value,
            },
            "output": {
                "label": output_label,
                "shape": output_shape,
                "preview": output_preview,
            },
        }

    def _edge(self, source, target, direction, label, shape, preview):
        return {
            "from": source,
            "to": target,
            "direction": direction,
            "packet": {
                "label": label,
                "shape": shape,
                "preview": preview,
            },
        }

    def _hidden_neuron_label(self, trace):
        shape = trace["forward_propagation"]["hidden_layer"]["shape"]
        return f"{shape.split(' x ')[-1]} neurons"

    def _weight_preview(self):
        return {
            "W1": self.model.W1[:2, :2].tolist(),
            "b1": self.model.b1[:4].tolist(),
            "W2": self.model.W2[:2, :4].tolist(),
            "b2": self.model.b2[:4].tolist(),
        }

    def _array_summary(self, values):
        if values.ndim == 1:
            shape = str(values.shape[0])
            preview = values[:6].tolist()
        else:
            shape = f"{values.shape[0]} x {values.shape[1]}"
            preview = values[:3, :6].tolist()
        return {
            "shape": shape,
            "preview": preview,
        }
