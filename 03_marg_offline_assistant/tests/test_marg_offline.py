import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.rag_engine import MargRagEngine
from backend.intent_model import IntentModel, load_intent_training_rows
from run_marg_desktop import (
    clean_answer_text,
    format_chat_answer,
    format_user_question,
    is_clear_question,
    run_self_test,
    should_focus_question_input_on_start,
)
from scripts.build_marg_app import build_script_app_bundle


ROOT = Path(__file__).resolve().parents[1]


class MargOfflineTests(unittest.TestCase):
    def test_loads_packaged_model_without_training_project_imports(self):
        engine = MargRagEngine.load(ROOT / "models" / "dpp-gita-rag-assistant-v2")

        self.assertEqual(engine.model_id, "dpp-gita-rag-assistant-v2")
        self.assertGreater(len(engine.vocabulary), 100)
        self.assertGreater(len(engine.verses), 100)
        self.assertEqual(engine.verse_embeddings.shape[1], 64)

    def test_answers_question_with_sources(self):
        engine = MargRagEngine.load(ROOT / "models" / "dpp-gita-rag-assistant-v2")

        response = engine.ask("How can I control anger?")

        self.assertEqual(response["model_id"], "dpp-gita-rag-assistant-v2")
        self.assertIn("answer", response)
        self.assertGreater(len(response["sources"]), 0)
        self.assertIn("chapter", response["sources"][0])

    def test_builds_mac_app_bundle_with_packaged_model(self):
        with TemporaryDirectory() as tmp:
            app_path = build_script_app_bundle(ROOT, Path(tmp))

            self.assertEqual(app_path.name, "Marg.app")
            self.assertTrue((app_path / "Contents" / "Info.plist").exists())
            self.assertTrue((app_path / "Contents" / "MacOS" / "Marg").exists())
            self.assertTrue((app_path / "Contents" / "Resources" / "app" / "run_marg_desktop.py").exists())
            self.assertTrue(
                (
                    app_path
                    / "Contents"
                    / "Resources"
                    / "app"
                    / "models"
                    / "dpp-gita-rag-assistant-v2"
                    / "verse_embeddings.npy"
                ).exists()
            )

    def test_script_bundle_is_not_marked_standalone(self):
        with TemporaryDirectory() as tmp:
            app_path = build_script_app_bundle(ROOT, Path(tmp))

            launcher = (app_path / "Contents" / "MacOS" / "Marg").read_text(encoding="utf-8")

            self.assertIn("/usr/bin/env python3", launcher)
            self.assertFalse((app_path / "Contents" / "Resources" / "standalone-runtime.txt").exists())

    def test_desktop_self_test_loads_model_and_answers(self):
        output = run_self_test("How can I control anger?")

        self.assertIn("Marg self-test OK", output)
        self.assertIn("Chapter", output)

    def test_chat_answer_format_is_readable_without_sources_panel(self):
        response = {
            "answer": "Chapter 2, Verse 58 gives the main direction: Control the senses. In daily life, apply this by pausing.",
            "sources": [
                {
                    "reference": "Chapter 2, Verse 58",
                    "translation": "Control the senses.",
                }
            ],
        }

        formatted = format_chat_answer(response)

        self.assertIn("According to Bhagavad Gita", formatted)
        self.assertIn("Chapter 2, Verse 58", formatted)
        self.assertIn("Control the senses.", formatted)
        self.assertNotIn("Main idea:", formatted)
        self.assertNotIn("Daily practice:", formatted)
        self.assertNotIn("score=", formatted)

    def test_user_question_format_is_compact_for_chat_history(self):
        formatted = format_user_question("How can I control anger?")

        self.assertEqual(formatted, "You\nHow can I control anger?")

    def test_unclear_input_is_rejected_before_retrieval_answer(self):
        self.assertFalse(is_clear_question("hii hgjhgjh jbk"))
        self.assertFalse(is_clear_question("cxbascbjasnvavcvdasbvcbads"))
        self.assertTrue(is_clear_question("How can I control anger?"))
        self.assertTrue(is_clear_question("Why should I earn money?"))
        self.assertTrue(is_clear_question("How to manage ego?"))
        self.assertTrue(is_clear_question("How to manage Team?"))
        self.assertTrue(is_clear_question("who am i?"))

    def test_clean_answer_hides_internal_related_support_sentence(self):
        answer = (
            "Chapter 2, Verse 58 gives the main direction: Control the senses. "
            "In daily life, apply this by pausing. "
            "Related support also appears in Chapter 5, Verse 23."
        )

        cleaned = clean_answer_text(answer)

        self.assertIn("According to Bhagavad Gita Chapter 2, Verse 58:", cleaned)
        self.assertNotIn("gives the main direction", cleaned)
        self.assertNotIn("Related support", cleaned)

    def test_answers_are_topic_specific_not_same_generic_line(self):
        engine = MargRagEngine.load(ROOT / "models" / "dpp-gita-rag-assistant-v2")

        anger = engine.ask("How can I control anger?")["answer"]
        earning = engine.ask("Why should I earn money?")["answer"]

        self.assertIn("anger", anger.lower())
        self.assertIn("work", earning.lower())
        self.assertNotEqual(anger, earning)

    def test_intent_training_csv_is_available_and_balanced(self):
        rows = load_intent_training_rows(ROOT / "data" / "marg_intent_questions.csv")
        intents = {row["intent"] for row in rows}

        self.assertEqual(len(rows), 10000)
        self.assertIn("ego", intents)
        self.assertIn("team_leadership", intents)
        self.assertIn("self_identity", intents)

    def test_hard_intent_training_csv_is_available(self):
        rows = load_intent_training_rows(ROOT / "data" / "marg_intent_questions_hard_10k.csv")
        intents = {row["intent"] for row in rows}

        self.assertEqual(len(rows), 10000)
        self.assertIn("peace_stress", intents)
        self.assertIn("money_work", intents)
        self.assertIn("purpose", intents)

    def test_intent_model_predicts_user_language(self):
        model = IntentModel.load(ROOT / "models" / "dpp-marg-intent-small-v1")

        self.assertEqual(model.predict("How to manage ego?")["intent"], "ego")
        self.assertEqual(model.predict("How to manage Team?")["intent"], "team_leadership")
        self.assertEqual(model.predict("who am i?")["intent"], "self_identity")
        self.assertEqual(model.predict("how do i balance money and life on weekends tbh")["intent"], "money_work")
        self.assertEqual(model.predict("pls i am struggling with: how do i build discipline")["intent"], "discipline")

    def test_marg_uses_intent_model_for_questions_not_keyword_list(self):
        engine = MargRagEngine.load(ROOT / "models" / "dpp-gita-rag-assistant-v2")
        intent_model = IntentModel.load(ROOT / "models" / "dpp-marg-intent-small-v1")

        ego = engine.ask("How to manage ego?", intent_model=intent_model)
        team = engine.ask("How to manage Team?", intent_model=intent_model)

        self.assertEqual(ego["intent"], "ego")
        self.assertEqual(team["intent"], "team_leadership")
        self.assertIn("ego", ego["answer"].lower())
        self.assertIn("team", team["answer"].lower())

    def test_intent_improves_self_identity_retrieval(self):
        engine = MargRagEngine.load(ROOT / "models" / "dpp-gita-rag-assistant-v2")
        intent_model = IntentModel.load(ROOT / "models" / "dpp-marg-intent-small-v1")

        response = engine.ask("who am i?", intent_model=intent_model)

        self.assertEqual(response["intent"], "self_identity")
        self.assertIn(response["sources"][0]["reference"], {"Chapter 2, Verse 13", "Chapter 2, Verse 20", "Chapter 2, Verse 22"})

    def test_core_gita_concept_questions_route_to_relevant_answers(self):
        engine = MargRagEngine.load(ROOT / "models" / "dpp-gita-rag-assistant-v2")
        intent_model = IntentModel.load(ROOT / "models" / "dpp-marg-intent-small-v1")

        worship = engine.ask("whom we need to worship?", intent_model=intent_model)
        atma = engine.ask("what is atma and parmatma?", intent_model=intent_model)
        god = engine.ask("am i a god?", intent_model=intent_model)

        self.assertEqual(worship["intent"], "devotion_worship")
        self.assertIn(worship["sources"][0]["reference"], {"Chapter 9, Verse 22", "Chapter 9, Verse 34", "Chapter 18, Verse 66"})
        self.assertIn("worship", worship["answer"].lower())

        self.assertEqual(atma["intent"], "atma_paramatma")
        self.assertIn(atma["sources"][0]["reference"], {"Chapter 2, Verse 20", "Chapter 13, Verse 23", "Chapter 15, Verse 15"})
        self.assertIn("atma", atma["answer"].lower())

        self.assertEqual(god["intent"], "self_identity")
        self.assertNotIn("please ask a clear", format_chat_answer(god).lower())
        self.assertIn("not the body", god["answer"].lower())

    def test_krishna_and_god_identity_questions_use_direct_sources(self):
        engine = MargRagEngine.load(ROOT / "models" / "dpp-gita-rag-assistant-v2")
        intent_model = IntentModel.load(ROOT / "models" / "dpp-marg-intent-small-v1")

        krishna = engine.ask("who is Krishna?", intent_model=intent_model)
        god = engine.ask("am I god?", intent_model=intent_model)

        self.assertEqual(krishna["intent"], "krishna_identity")
        self.assertIn(krishna["sources"][0]["reference"], {"Chapter 10, Verse 8", "Chapter 7, Verse 7", "Chapter 7, Verse 19"})
        self.assertIn("Krishna", krishna["answer"])
        self.assertIn(god["sources"][0]["reference"], {"Chapter 2, Verse 13", "Chapter 2, Verse 20", "Chapter 2, Verse 22"})

    def test_preferred_verses_artifact_is_built_from_training_csv(self):
        artifact = ROOT / "models" / "dpp-marg-intent-small-v1" / "intent_verse_preferences.json"

        preferences = json.loads(artifact.read_text(encoding="utf-8"))

        self.assertIn("devotion_worship", preferences)
        self.assertIn("atma_paramatma", preferences)
        self.assertIn("9.22", preferences["devotion_worship"])
        self.assertIn("2.20", preferences["atma_paramatma"])

    def test_retrieval_uses_csv_preferred_verses_for_boosting(self):
        engine = MargRagEngine.load(ROOT / "models" / "dpp-gita-rag-assistant-v2")
        intent_model = IntentModel.load(ROOT / "models" / "dpp-marg-intent-small-v1")

        response = engine.ask("who is Krishna?", intent_model=intent_model)

        self.assertEqual(response["boost_source"], "intent_verse_preferences")
        self.assertIn(response["sources"][0]["reference"], {"Chapter 10, Verse 8", "Chapter 7, Verse 7", "Chapter 7, Verse 19"})

    def test_short_spiritual_questions_are_clear(self):
        self.assertTrue(is_clear_question("am i a god?"))
        self.assertTrue(is_clear_question("what is atma?"))
        self.assertTrue(is_clear_question("who is krishna?"))

    def test_desktop_starts_with_question_input_focused(self):
        self.assertTrue(should_focus_question_input_on_start())


if __name__ == "__main__":
    unittest.main()
