import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset import GitaVerse
from src.embedding.model import EmbeddingModel, cosine_scores
from src.embedding.search import EmbeddingSearchIndex, search_embeddings
from src.embedding.storage import load_embedding_artifacts, save_embedding_artifacts
from src.embedding.training_data import TrainingPair, build_training_examples
from src.embedding.vocabulary import build_embedding_vocabulary, encode_text


class GitaEmbeddingModelTests(unittest.TestCase):
    def sample_verses(self):
        return [
            GitaVerse("5", "23", "", "Control desire and anger through sense discipline.", "Anger becomes calm through practice.", "anger"),
            GitaVerse("6", "26", "", "Bring the restless mind back under the control of the self.", "Mind is trained by repeated practice.", "mind"),
            GitaVerse("3", "19", "", "Act as a matter of duty without attachment to fruits.", "Duty purifies work.", "duty"),
        ]

    def sample_pairs(self):
        return [
            TrainingPair("How can I control anger?", "5", "23", "anger"),
            TrainingPair("How do I calm my restless mind?", "6", "26", "mind"),
            TrainingPair("What should I do about duty?", "3", "19", "duty"),
        ]

    def test_vocabulary_encodes_known_and_unknown_tokens(self):
        vocabulary = build_embedding_vocabulary(["control anger", "restless mind"], max_size=5)

        encoded = encode_text("control unknown", vocabulary, max_length=4)

        self.assertEqual(encoded[0], vocabulary["control"])
        self.assertEqual(encoded[1], vocabulary["<UNK>"])
        self.assertEqual(len(encoded), 4)

    def test_training_examples_link_question_positive_and_negative(self):
        examples = build_training_examples(self.sample_pairs(), self.sample_verses())

        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0].positive_verse.chapter, "5")
        self.assertNotEqual(examples[0].negative_verse.verse, examples[0].positive_verse.verse)

    def test_embedding_model_outputs_normalized_vectors(self):
        model = EmbeddingModel.create(vocab_size=20, token_dim=8, output_dim=6, seed=7)

        vector = model.embed_token_ids([1, 2, 3, 0])

        self.assertEqual(vector.shape, (6,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_embedding_search_returns_nearest_verse(self):
        verses = self.sample_verses()
        vocabulary = build_embedding_vocabulary([verse.searchable_text() for verse in verses] + ["control anger"], max_size=100)
        model = EmbeddingModel.create(vocab_size=len(vocabulary), token_dim=8, output_dim=6, seed=2)
        verse_embeddings = np.vstack([
            model.embed_text(verse.searchable_text(), vocabulary, max_length=20) for verse in verses
        ])
        question_embedding = verse_embeddings[0]
        index = EmbeddingSearchIndex(
            model_id="dpp-gita-embedding-small-v1",
            model=model,
            vocabulary=vocabulary,
            verses=verses,
            verse_embeddings=verse_embeddings,
            max_length=20,
        )

        results = search_embeddings(index, "control anger", top_k=1, query_vector=question_embedding)

        self.assertEqual(results[0].verse.chapter, "5")
        self.assertEqual(results[0].verse.verse, "23")
        self.assertEqual(results[0].matched_words, [])

    def test_embedding_artifacts_round_trip(self):
        verses = self.sample_verses()
        vocabulary = build_embedding_vocabulary([verse.searchable_text() for verse in verses], max_size=100)
        model = EmbeddingModel.create(vocab_size=len(vocabulary), token_dim=8, output_dim=6, seed=3)
        verse_embeddings = np.vstack([
            model.embed_text(verse.searchable_text(), vocabulary, max_length=20) for verse in verses
        ])

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "dpp-gita-embedding-small-v1"
            save_embedding_artifacts(
                model_dir=model_dir,
                model_id="dpp-gita-embedding-small-v1",
                model=model,
                vocabulary=vocabulary,
                verses=verses,
                verse_embeddings=verse_embeddings,
                config={"max_length": 20},
                metrics={"loss": 0.5},
            )

            loaded = load_embedding_artifacts(model_dir)

        self.assertEqual(loaded.model_id, "dpp-gita-embedding-small-v1")
        self.assertEqual(loaded.verse_embeddings.shape, (3, 6))
        self.assertIn("control", loaded.vocabulary)

    def test_cosine_scores_orders_closest_vector_first(self):
        query = np.array([1.0, 0.0])
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]])

        scores = cosine_scores(query, matrix)

        self.assertGreater(scores[1], scores[0])


if __name__ == "__main__":
    unittest.main()
