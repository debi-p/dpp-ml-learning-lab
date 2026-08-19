import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_verses_csv


def validate_dataset(path):
    verses = load_verses_csv(path)
    chapters = Counter(verse.chapter for verse in verses)
    duplicates = Counter((verse.chapter, verse.verse) for verse in verses)
    duplicate_rows = [key for key, count in duplicates.items() if count > 1]
    missing_translation = [verse for verse in verses if not verse.translation]
    missing_commentary = [verse for verse in verses if not verse.commentary]

    return {
        "row_count": len(verses),
        "chapter_count": len(chapters),
        "chapters": dict(sorted(chapters.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999)),
        "duplicate_count": len(duplicate_rows),
        "missing_translation_count": len(missing_translation),
        "missing_commentary_count": len(missing_commentary),
        "sample": verses[:3],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate structured Gita dataset.")
    parser.add_argument("--input", default="data/gita_verses.csv")
    args = parser.parse_args()

    report = validate_dataset(args.input)
    print(f"Rows: {report['row_count']}")
    print(f"Chapters: {report['chapter_count']}")
    print(f"Duplicate chapter/verse rows: {report['duplicate_count']}")
    print(f"Missing translations: {report['missing_translation_count']}")
    print(f"Missing commentaries: {report['missing_commentary_count']}")
    print("Rows by chapter:")
    for chapter, count in report["chapters"].items():
        print(f"  Chapter {chapter}: {count}")
    print("Sample rows:")
    for verse in report["sample"]:
        print(f"  {verse.chapter}.{verse.verse}: {verse.translation[:120]}")


if __name__ == "__main__":
    main()
