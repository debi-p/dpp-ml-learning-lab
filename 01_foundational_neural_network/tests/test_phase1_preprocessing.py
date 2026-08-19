import csv
import tempfile
import unittest
from pathlib import Path

from src.data import load_messages
from src.features import build_vocabulary, vectorize_tokens
from src.labels import build_label
from src.text import clean_text, tokenize


class Phase1PreprocessingTests(unittest.TestCase):
    def test_load_messages_reads_category_and_message_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Category", "Message"])
                writer.writeheader()
                writer.writerow({"Category": "ham", "Message": "Can we meet tomorrow?"})
                writer.writerow({"Category": "spam", "Message": "Free prize claim now"})

            rows = load_messages(csv_path)

        self.assertEqual(
            rows,
            [
                {"category": "ham", "message": "Can we meet tomorrow?"},
                {"category": "spam", "message": "Free prize claim now"},
            ],
        )

    def test_build_label_creates_four_target_classes(self):
        self.assertEqual(build_label("spam", "Free prize claim now"), "spam")
        self.assertEqual(build_label("ham", "Project review meeting tomorrow"), "work")
        self.assertEqual(build_label("ham", "Discount voucher available"), "promotion")
        self.assertEqual(build_label("ham", "Are you coming home tonight"), "personal")

    def test_build_label_prefers_promotion_when_ham_has_promotion_and_work_words(self):
        self.assertEqual(build_label("ham", "Discount voucher offer available today"), "promotion")

    def test_clean_text_and_tokenize_normalize_message(self):
        cleaned = clean_text("Can we REVIEW the project tomorrow?!")
        tokens = tokenize(cleaned)

        self.assertEqual(cleaned, "can we review the project tomorrow")
        self.assertEqual(tokens, ["can", "we", "review", "the", "project", "tomorrow"])

    def test_build_vocabulary_keeps_most_common_words(self):
        tokenized_messages = [
            ["free", "prize", "free"],
            ["project", "meeting"],
            ["free", "meeting"],
        ]

        vocabulary = build_vocabulary(tokenized_messages, max_size=3)

        self.assertEqual(vocabulary, {"free": 0, "meeting": 1, "prize": 2})

    def test_vectorize_tokens_creates_bag_of_words_counts(self):
        vocabulary = {"free": 0, "meeting": 1, "project": 2}

        vector = vectorize_tokens(["free", "free", "project", "unknown"], vocabulary)

        self.assertEqual(vector.tolist(), [2.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
