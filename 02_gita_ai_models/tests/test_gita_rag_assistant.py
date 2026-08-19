import unittest

from src.dataset import GitaVerse
from src.embedding.search import EmbeddingSearchResult
from src.rag.answer_builder import build_rag_answer
from src.rag.context_builder import build_augmented_context
from src.rag.transformer_bridge import build_transformer_prompt, build_rag_transformer_answer


class GitaRagAssistantTests(unittest.TestCase):
    def sample_results(self):
        return [
            EmbeddingSearchResult(
                verse=GitaVerse(
                    "5",
                    "23",
                    "",
                    "Before giving up this present body, if one is able to tolerate the urges of the material senses and check the force of desire and anger, he is a yogi and is happy in this world.",
                    "Material desires, when unsatiated, generate anger, and thus the mind becomes agitated. One must practice to control them.",
                    "anger;desire;control",
                ),
                score=0.91,
                matched_words=[],
            ),
            EmbeddingSearchResult(
                verse=GitaVerse(
                    "6",
                    "26",
                    "",
                    "Wherever the restless mind wanders, one should bring it back under the control of the self.",
                    "A disciplined mind becomes peaceful through repeated practice.",
                    "mind;practice;peace",
                ),
                score=0.82,
                matched_words=[],
            ),
        ]

    def test_context_builder_keeps_question_and_sources(self):
        context = build_augmented_context("How can I control anger?", self.sample_results())

        self.assertEqual(context["question"], "How can I control anger?")
        self.assertEqual(len(context["sources"]), 2)
        self.assertEqual(context["sources"][0]["reference"], "Chapter 5, Verse 23")
        self.assertLessEqual(len(context["sources"][0]["translation"]), 260)

    def test_rag_answer_is_concise_and_source_backed(self):
        context = build_augmented_context("How can I control anger?", self.sample_results())

        response = build_rag_answer(context)

        self.assertEqual(response["model_id"], "dpp-gita-rag-assistant-v2")
        self.assertIn("Chapter 5, Verse 23", response["answer"])
        self.assertIn("daily life", response["answer"].lower())
        self.assertLess(len(response["answer"]), 900)
        self.assertEqual(response["sources"][0]["chapter"], "5")

    def test_rag_answer_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            build_augmented_context("", self.sample_results())

    def test_transformer_prompt_contains_question_and_sources(self):
        context = build_augmented_context("How can I control anger?", self.sample_results())

        prompt = build_transformer_prompt(context)

        self.assertIn("Question: How can I control anger?", prompt)
        self.assertIn("Chapter 5, Verse 23", prompt)
        self.assertIn("Answer:", prompt)

    def test_rag_transformer_answer_keeps_sources_and_generation_trace(self):
        context = build_augmented_context("How can I control anger?", self.sample_results())

        response = build_rag_transformer_answer(
            context,
            generated_text="practice steady control",
            generation_steps=[{"next_token": "practice"}],
        )

        self.assertEqual(response["model_id"], "dpp-gita-rag-transformer-v1")
        self.assertEqual(response["question"], "How can I control anger?")
        self.assertIn("practice", response["answer"])
        self.assertEqual(response["sources"][0]["chapter"], "5")
        self.assertEqual(response["generation_steps"][0]["next_token"], "practice")


if __name__ == "__main__":
    unittest.main()
