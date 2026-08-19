import tempfile
import unittest
from pathlib import Path

from sdk.classifier import EmailClassifier
from src.model import NeuralNetwork
from src.storage import save_artifacts


class Phase1SdkTests(unittest.TestCase):
    def test_email_classifier_loads_model_and_predicts_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_id = "test-model"
            models_dir = Path(tmpdir)
            model_dir = models_dir / model_id
            model = NeuralNetwork(input_size=3, hidden_size=2, output_size=4, seed=3)
            save_artifacts(
                model_dir,
                model,
                {"project": 0, "home": 1, "free": 2},
                ["work", "personal", "promotion", "spam"],
                {"model_id": model_id},
            )

            classifier = EmailClassifier.load(model_id, models_dir=models_dir)
            result = classifier.predict("project meeting")

        self.assertEqual(result["model_id"], model_id)
        self.assertIn(result["prediction"], ["work", "personal", "promotion", "spam"])
        self.assertEqual(set(result["confidence"]), {"work", "personal", "promotion", "spam"})

    def test_email_classifier_exposes_generic_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_id = "test-model"
            models_dir = Path(tmpdir)
            model_dir = models_dir / model_id
            model = NeuralNetwork(input_size=3, hidden_size=2, output_size=4, seed=3)
            save_artifacts(
                model_dir,
                model,
                {"project": 0, "home": 1, "free": 2},
                ["work", "personal", "promotion", "spam"],
                {"model_id": model_id, "task": "message_classification"},
            )

            classifier = EmailClassifier.load(model_id, models_dir=models_dir)
            metadata = classifier.metadata()

        self.assertEqual(metadata["model_id"], model_id)
        self.assertEqual(metadata["task_type"], "classification")
        self.assertEqual(metadata["target_type"], "class_label")
        self.assertEqual(metadata["labels"], ["work", "personal", "promotion", "spam"])

    def test_email_classifier_inspect_forward_exposes_pipeline_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_id = "test-model"
            models_dir = Path(tmpdir)
            model_dir = models_dir / model_id
            model = NeuralNetwork(input_size=3, hidden_size=2, output_size=4, seed=3)
            save_artifacts(
                model_dir,
                model,
                {"project": 0, "home": 1, "free": 2},
                ["work", "personal", "promotion", "spam"],
                {"model_id": model_id},
            )

            classifier = EmailClassifier.load(model_id, models_dir=models_dir)
            trace = classifier.inspect_forward("Project meeting at home")

        self.assertEqual(trace["model_id"], model_id)
        self.assertEqual(trace["input"]["cleaned"], "project meeting at home")
        self.assertEqual(trace["input"]["tokens"], ["project", "meeting", "at", "home"])
        self.assertEqual(trace["vectorization"]["non_zero_features"], {"project": 1.0, "home": 1.0})
        self.assertEqual(trace["vectorization"]["shape"], "1 x 3")
        self.assertEqual(trace["forward_propagation"]["hidden_layer"]["shape"], "1 x 2")
        self.assertEqual(trace["forward_propagation"]["activation"]["shape"], "1 x 2")
        self.assertEqual(trace["forward_propagation"]["output_layer"]["shape"], "1 x 4")
        self.assertEqual(set(trace["forward_propagation"]["softmax"]["probabilities"]), {"work", "personal", "promotion", "spam"})
        self.assertIn(trace["prediction"], ["work", "personal", "promotion", "spam"])
        self.assertEqual(trace["model_metadata"]["labels"], ["work", "personal", "promotion", "spam"])
        self.assertIn("summary", trace)
        self.assertIn("flow", trace)
        self.assertGreater(len(trace["flow"]["nodes"]), 0)
        self.assertGreater(len(trace["flow"]["edges"]), 0)
        self.assertGreater(len(trace["flow"]["timeline"]), 0)
        self.assertEqual(trace["flow"]["nodes"][0]["name"], "Input Message")

    def test_email_classifier_inspect_training_step_exposes_loss_gradients_and_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_id = "test-model"
            models_dir = Path(tmpdir)
            model_dir = models_dir / model_id
            model = NeuralNetwork(input_size=3, hidden_size=2, output_size=4, seed=3)
            save_artifacts(
                model_dir,
                model,
                {"project": 0, "home": 1, "free": 2},
                ["work", "personal", "promotion", "spam"],
                {"model_id": model_id},
            )

            classifier = EmailClassifier.load(model_id, models_dir=models_dir)
            trace = classifier.inspect_training_step("Project meeting at home", correct_label="work", learning_rate=0.1)

        self.assertEqual(trace["model_id"], model_id)
        self.assertEqual(trace["training"]["correct_label"], "work")
        self.assertGreater(trace["training"]["loss"], 0.0)
        self.assertEqual(trace["backpropagation"]["dW2"]["shape"], "2 x 4")
        self.assertEqual(trace["backpropagation"]["db2"]["shape"], "4")
        self.assertEqual(trace["backpropagation"]["dW1"]["shape"], "3 x 2")
        self.assertEqual(trace["backpropagation"]["db1"]["shape"], "2")
        self.assertIn("before", trace["weight_update"])
        self.assertIn("after", trace["weight_update"])
        self.assertIn("flow", trace)
        self.assertIn("loss_calculation", [node["id"] for node in trace["flow"]["nodes"]])
        self.assertIn("weight_update", [node["id"] for node in trace["flow"]["nodes"]])
        self.assertIn("training_step_complete", [node["id"] for node in trace["flow"]["nodes"]])
        self.assertNotIn(
            ("weight_update", "hidden_layer_1"),
            [(edge["from"], edge["to"]) for edge in trace["flow"]["edges"]],
        )
        self.assertIn(
            ("weight_update", "training_step_complete"),
            [(edge["from"], edge["to"]) for edge in trace["flow"]["edges"]],
        )


if __name__ == "__main__":
    unittest.main()
