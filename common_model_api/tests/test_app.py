import unittest

from fastapi.testclient import TestClient

from app import app


class ModelApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_models_returns_registered_models(self):
        response = self.client.get("/models")

        self.assertEqual(response.status_code, 200)
        models = response.json()
        self.assertGreaterEqual(len(models), 1)
        self.assertEqual(models[0]["model_id"], "dpp-email-classifier-small-v1")
        self.assertEqual(models[0]["target_type"], "class_label")
        self.assertEqual(models[0]["labels"], ["work", "personal", "promotion", "spam"])

    def test_predict_returns_prediction_and_confidence(self):
        response = self.client.post(
            "/predict",
            json={
                "model_id": "dpp-email-classifier-small-v1",
                "input": "Can we review the project deadline tomorrow?",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], "dpp-email-classifier-small-v1")
        self.assertIn(body["prediction"], ["work", "personal", "promotion", "spam"])
        self.assertEqual(set(body["confidence"]), {"work", "personal", "promotion", "spam"})

    def test_predict_returns_gita_rag_answer_and_sources(self):
        response = self.client.post(
            "/predict",
            json={
                "model_id": "dpp-gita-rag-assistant-v2",
                "input": "How can I control anger?",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], "dpp-gita-rag-assistant-v2")
        self.assertEqual(body["retriever_model_id"], "dpp-gita-embedding-small-v1")
        self.assertIn("answer", body)
        self.assertGreater(len(body["sources"]), 0)
        self.assertIn("chapter", body["sources"][0])

    def test_inspect_rag_returns_retrieval_and_context_trace(self):
        response = self.client.post(
            "/inspect-rag",
            json={
                "model_id": "dpp-gita-rag-assistant-v2",
                "input": "How can I control anger?",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], "dpp-gita-rag-assistant-v2")
        self.assertIn("retrieval", body)
        self.assertIn("augmented_context", body)
        self.assertIn("answer", body)
        self.assertGreater(len(body["retrieval"]["results"]), 0)

    def test_predict_returns_gita_transformer_generation(self):
        response = self.client.post(
            "/predict",
            json={
                "model_id": "dpp-gita-tiny-transformer-v1",
                "input": "control anger",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], "dpp-gita-tiny-transformer-v1")
        self.assertEqual(body["prompt"], "control anger")
        self.assertIn("generated_text", body)
        self.assertIn("generated_tokens", body)
        self.assertGreater(len(body["generated_tokens"]), 0)

    def test_inspect_transformer_returns_generation_trace(self):
        response = self.client.post(
            "/inspect-transformer",
            json={
                "model_id": "dpp-gita-tiny-transformer-v1",
                "input": "control anger",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], "dpp-gita-tiny-transformer-v1")
        self.assertIn("tokenization", body)
        self.assertIn("generation_steps", body)
        self.assertIn("model_config", body)
        self.assertGreater(len(body["generation_steps"]), 0)

    def test_predict_returns_gita_rag_transformer_answer(self):
        response = self.client.post(
            "/predict",
            json={
                "model_id": "dpp-gita-rag-transformer-v1",
                "input": "How can I control anger?",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], "dpp-gita-rag-transformer-v1")
        self.assertEqual(body["retriever_model_id"], "dpp-gita-embedding-small-v1")
        self.assertEqual(body["generator_model_id"], "dpp-gita-tiny-transformer-v1")
        self.assertIn("answer", body)
        self.assertGreater(len(body["sources"]), 0)
        self.assertIn("generation_steps", body)

    def test_inspect_rag_transformer_returns_full_combined_trace(self):
        response = self.client.post(
            "/inspect-rag-transformer",
            json={
                "model_id": "dpp-gita-rag-transformer-v1",
                "input": "How can I control anger?",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], "dpp-gita-rag-transformer-v1")
        self.assertIn("retrieval", body)
        self.assertIn("augmented_context", body)
        self.assertIn("transformer_prompt", body)
        self.assertIn("generation_steps", body)

    def test_inspect_forward_returns_pipeline_trace(self):
        response = self.client.post(
            "/inspect-forward",
            json={
                "model_id": "dpp-email-classifier-small-v1",
                "input": "Can we review the project deadline tomorrow?",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], "dpp-email-classifier-small-v1")
        self.assertIn("tokens", body["input"])
        self.assertIn("vectorization", body)
        self.assertIn("forward_propagation", body)
        self.assertIn("model_metadata", body)
        self.assertIn("summary", body)
        self.assertIn("flow", body)
        self.assertGreater(len(body["flow"]["nodes"]), 0)
        self.assertGreater(len(body["flow"]["edges"]), 0)

    def test_inspect_training_step_returns_backpropagation_trace(self):
        response = self.client.post(
            "/inspect-training-step",
            json={
                "model_id": "dpp-email-classifier-small-v1",
                "input": "Can we review the project deadline tomorrow?",
                "correct_label": "work",
                "learning_rate": 0.1,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["training"]["correct_label"], "work")
        self.assertIn("backpropagation", body)
        self.assertIn("weight_update", body)
        self.assertIn("loss_calculation", [node["id"] for node in body["flow"]["nodes"]])

    def test_unknown_model_returns_404(self):
        response = self.client.post(
            "/predict",
            json={"model_id": "missing-model", "input": "hello"},
        )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
