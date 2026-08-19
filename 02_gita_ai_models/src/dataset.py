import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from src.text import normalize_for_search


@dataclass
class GitaVerse:
    chapter: str
    verse: str
    sanskrit: str
    translation: str
    commentary: str
    tags: str

    def searchable_text(self):
        return normalize_for_search(" ".join([self.translation, self.commentary, self.tags]).strip())


FIELDNAMES = ["chapter", "verse", "sanskrit", "translation", "commentary", "tags"]


def load_verses_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            GitaVerse(
                chapter=(row.get("chapter") or "").strip(),
                verse=(row.get("verse") or "").strip(),
                sanskrit=(row.get("sanskrit") or "").strip(),
                translation=(row.get("translation") or "").strip(),
                commentary=(row.get("commentary") or "").strip(),
                tags=(row.get("tags") or "").strip(),
            )
            for row in rows
        ]


def save_verses_csv(verses, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for verse in verses:
            writer.writerow(asdict(verse))
