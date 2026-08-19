import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingPair:
    question: str
    positive_chapter: str
    positive_verse: str
    topic: str
    answer: str = ""


@dataclass
class TrainingExample:
    question: str
    positive_verse: object
    negative_verse: object
    topic: str


def load_training_pairs(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            TrainingPair(
                question=(row.get("question") or "").strip(),
                answer=(row.get("answer") or "").strip(),
                positive_chapter=(row.get("positive_chapter") or "").strip(),
                positive_verse=(row.get("positive_verse") or "").strip(),
                topic=(row.get("topic") or "").strip(),
            )
            for row in reader
        ]


def build_verse_lookup(verses):
    return {(verse.chapter, verse.verse): verse for verse in verses}


def build_training_examples(pairs, verses, seed=13):
    rng = random.Random(seed)
    lookup = build_verse_lookup(verses)
    examples = []

    for pair in pairs:
        positive = lookup.get((pair.positive_chapter, pair.positive_verse))
        if positive is None:
            continue

        candidates = [
            verse
            for verse in verses
            if (verse.chapter, verse.verse) != (positive.chapter, positive.verse)
            and pair.topic.lower() not in (verse.tags or "").lower()
        ]
        if not candidates:
            candidates = [verse for verse in verses if (verse.chapter, verse.verse) != (positive.chapter, positive.verse)]
        if not candidates:
            continue

        examples.append(
            TrainingExample(
                question=pair.question,
                positive_verse=positive,
                negative_verse=rng.choice(candidates),
                topic=pair.topic,
            )
        )

    return examples


def validate_pairs_against_verses(pairs, verses):
    lookup = build_verse_lookup(verses)
    matched = []
    unmatched = []
    for pair in pairs:
        key = (pair.positive_chapter, pair.positive_verse)
        if key in lookup:
            matched.append(pair)
        else:
            unmatched.append(pair)
    return matched, unmatched

