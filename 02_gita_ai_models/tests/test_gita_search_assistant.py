import tempfile
import unittest
from pathlib import Path

from src.answer_builder import build_answer
from src.dataset import GitaVerse, save_verses_csv
from scripts.build_gita_dataset import parse_verses
from src.retrieval import build_search_model, search
from src.storage import load_search_model, save_search_model
from src.text import normalize_for_display, normalize_for_search, tokenize
from sdk.gita_assistant import GitaAssistant


class GitaSearchAssistantTests(unittest.TestCase):
    def sample_verses(self):
        return [
            GitaVerse(
                chapter="2",
                verse="62-63",
                sanskrit="",
                translation="While contemplating the objects of the senses, attachment develops. From attachment desire arises, and from desire anger arises.",
                commentary="Anger grows when desire is blocked. The mind becomes disturbed and judgment is lost.",
                tags="anger;desire;attachment",
            ),
            GitaVerse(
                chapter="3",
                verse="19",
                sanskrit="",
                translation="One should act as a matter of duty, without attachment to the fruits of work.",
                commentary="Duty done without selfish attachment purifies the heart.",
                tags="duty;work;karma",
            ),
            GitaVerse(
                chapter="6",
                verse="26",
                sanskrit="",
                translation="Wherever the restless mind wanders, one should bring it back under the control of the self.",
                commentary="A disciplined mind becomes peaceful through repeated practice.",
                tags="mind;peace;discipline",
            ),
        ]

    def test_tokenize_normalizes_words(self):
        self.assertEqual(
            tokenize("How can I control anger?"),
            ["how", "can", "i", "control", "anger"],
        )

    def test_normalize_for_search_removes_diacritics(self):
        text = "Bhagavad-gétä Kåñëa yogé gosvämé svämé jïäna"

        normalized = normalize_for_search(text)

        self.assertIn("bhagavad gita", normalized)
        self.assertIn("krishna", normalized)
        self.assertIn("yogi", normalized)
        self.assertIn("gosvami", normalized)
        self.assertIn("svami", normalized)
        self.assertIn("jnana", normalized)

    def test_normalize_for_display_makes_answer_readable(self):
        text = "He is a yogé and is called gosvämé, or svämé. Kåñëa teaches jïäna."

        normalized = normalize_for_display(text)

        self.assertIn("yogi", normalized)
        self.assertIn("gosvami", normalized)
        self.assertIn("svami", normalized)
        self.assertIn("Krishna", normalized)
        self.assertNotIn("yogé", normalized)

    def test_search_returns_relevant_anger_verse(self):
        model = build_search_model(self.sample_verses())

        results = search(model, "How can I control anger?", top_k=2)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].verse.chapter, "2")
        self.assertEqual(results[0].verse.verse, "62-63")
        self.assertGreater(results[0].score, 0)

    def test_answer_builder_returns_answer_with_sources(self):
        verses = [
            GitaVerse(
                chapter="5",
                verse="23",
                sanskrit="",
                translation="He is a yogé and is happy in this world.",
                commentary="Such a person is called gosvämé, or svämé.",
                tags="anger",
            )
        ]
        model = build_search_model(verses)
        results = search(model, "How can I control anger?", top_k=2)

        answer = build_answer("How can I control anger?", results)

        self.assertIn("answer", answer)
        self.assertIn("sources", answer)
        self.assertIn("Chapter 5, Verse 23", answer["answer"])
        self.assertIn("yogi", answer["answer"])
        self.assertIn("gosvami", answer["answer"])
        self.assertNotIn("yogé", answer["answer"])
        self.assertEqual(answer["sources"][0]["chapter"], "5")

    def test_sdk_loads_saved_model_and_answers_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "dpp-gita-search-assistant-v1"
            dataset_path = root / "data" / "gita_verses.csv"
            dataset_path.parent.mkdir(parents=True)
            save_verses_csv(self.sample_verses(), dataset_path)

            model = build_search_model(self.sample_verses(), model_id="dpp-gita-search-assistant-v1")
            save_search_model(model, model_dir)

            loaded = load_search_model(model_dir)
            assistant = GitaAssistant(loaded)
            response = assistant.ask("How can I control anger?")

            self.assertEqual(response["model_id"], "dpp-gita-search-assistant-v1")
            self.assertEqual(response["sources"][0]["chapter"], "2")
            self.assertIn("anger", response["answer"].lower())

    def test_parse_verses_expands_grouped_text_ranges(self):
        raw_text = """
CHAPTER TWELVE
TEXTS 13-14
some sanskrit
TRANSLATION
One who is not envious but is a kind friend to all living entities is dear to Me.
PURPORT
These verses describe qualities of a pure devotee.
TEXT 15
TRANSLATION
He who does not disturb anyone is dear to Me.
PURPORT
Another quality is steadiness.
"""

        verses = parse_verses(raw_text)
        keys = {(verse.chapter, verse.verse) for verse in verses}

        self.assertIn(("12", "13"), keys)
        self.assertIn(("12", "14"), keys)
        self.assertIn(("12", "15"), keys)


if __name__ == "__main__":
    unittest.main()
