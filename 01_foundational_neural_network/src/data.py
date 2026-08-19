import csv
from pathlib import Path


def load_messages(csv_path):
    path = Path(csv_path)
    rows = []

    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = (row.get("Category") or "").strip().lower()
            message = (row.get("Message") or "").strip()
            if category and message:
                rows.append({"category": category, "message": message})

    return rows
