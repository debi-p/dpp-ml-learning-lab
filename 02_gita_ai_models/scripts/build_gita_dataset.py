import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import GitaVerse, save_verses_csv
from src.text import clean_text


CHAPTER_NUMBERS = {
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
    "TEN": "10",
    "ELEVEN": "11",
    "TWELVE": "12",
    "THIRTEEN": "13",
    "FOURTEEN": "14",
    "FIFTEEN": "15",
    "SIXTEEN": "16",
    "SEVENTEEN": "17",
    "EIGHTEEN": "18",
}


def strip_page_markers(text):
    text = re.sub(r"\n--- PAGE \d+ ---\n", "\n", text)
    text = re.sub(r"For more free downloadable original books visit\s+www\.Krishnapath\.org", " ", text)
    return text


def chapter_for_position(text, position):
    chapters = list(re.finditer(r"\bCHAPTER\s+([A-Z]+)\b", text))
    current = ""
    for match in chapters:
        if match.start() <= position:
            current = CHAPTER_NUMBERS.get(match.group(1), current)
        else:
            break
    return current


def section_between(block, start_marker, end_markers):
    start = re.search(rf"\b{re.escape(start_marker)}\b", block)
    if not start:
        return ""

    rest = block[start.end() :]
    end_positions = []
    for marker in end_markers:
        found = re.search(rf"\b{re.escape(marker)}\b", rest)
        if found:
            end_positions.append(found.start())

    end = min(end_positions) if end_positions else len(rest)
    return clean_text(rest[:end])


def expand_verse_reference(reference):
    normalized = reference.replace("–", "-").replace("—", "-")
    if "-" not in normalized:
        return [normalized]

    start_text, end_text = normalized.split("-", 1)
    start = int(start_text)
    end = int(end_text)
    return [str(number) for number in range(start, end + 1)]


def parse_verses(raw_text):
    text = strip_page_markers(raw_text)
    matches = list(re.finditer(r"\bTEXTS?\s+(\d+(?:\s*[-–—]\s*\d+)?)\b", text))
    verses = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        chapter = chapter_for_position(text, match.start())
        verse_numbers = expand_verse_reference(re.sub(r"\s+", "", match.group(1)))
        translation = section_between(block, "TRANSLATION", ["PURPORT", "TEXT"])
        commentary = section_between(block, "PURPORT", ["TEXT"])

        if not chapter or not translation:
            continue

        for verse_number in verse_numbers:
            verses.append(
                GitaVerse(
                    chapter=chapter,
                    verse=verse_number,
                    sanskrit="",
                    translation=translation,
                    commentary=commentary,
                    tags="",
                )
            )

    return verses


def build_dataset(raw_text_path, output_csv_path, clean_text_path=None):
    raw_text = Path(raw_text_path).read_text(encoding="utf-8")
    cleaned = strip_page_markers(raw_text)
    if clean_text_path:
        clean_output = Path(clean_text_path)
        clean_output.parent.mkdir(parents=True, exist_ok=True)
        clean_output.write_text(cleaned, encoding="utf-8")

    verses = parse_verses(raw_text)
    save_verses_csv(verses, output_csv_path)
    return verses


def main():
    parser = argparse.ArgumentParser(description="Build structured Gita CSV dataset from extracted PDF text.")
    parser.add_argument("--input", default="data/extracted_raw_text.txt")
    parser.add_argument("--output", default="data/gita_verses.csv")
    parser.add_argument("--clean-output", default="data/gita_clean_text.txt")
    args = parser.parse_args()

    verses = build_dataset(args.input, args.output, clean_text_path=args.clean_output)
    print(f"Wrote {len(verses)} verse rows to {args.output}")
    print(f"Wrote cleaned text to {args.clean_output}")


if __name__ == "__main__":
    main()
