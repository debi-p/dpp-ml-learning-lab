import json
from dataclasses import asdict
from pathlib import Path

from src.dataset import GitaVerse
from src.retrieval import SearchModel


def save_search_model(model, model_dir):
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "model_card.json").write_text(
        json.dumps(
            {
                "model_id": model.model_id,
                "type": "gita_search_assistant_from_scratch",
                "vocabulary_size": len(model.vocabulary),
                "verse_count": len(model.verses),
                "algorithm": "TF-IDF + cosine similarity + template answer builder",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "vocabulary.json").write_text(json.dumps(model.vocabulary, indent=2), encoding="utf-8")
    (output / "idf.json").write_text(json.dumps(model.idf, indent=2), encoding="utf-8")
    (output / "verse_index.json").write_text(
        json.dumps([asdict(verse) for verse in model.verses], indent=2),
        encoding="utf-8",
    )


def load_search_model(model_dir):
    root = Path(model_dir)
    card = json.loads((root / "model_card.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    idf = json.loads((root / "idf.json").read_text(encoding="utf-8"))
    verse_rows = json.loads((root / "verse_index.json").read_text(encoding="utf-8"))
    verses = [GitaVerse(**row) for row in verse_rows]

    from src.vectorize import vectorize_tfidf

    verse_vectors = [vectorize_tfidf(verse.searchable_text(), vocabulary, idf) for verse in verses]
    return SearchModel(
        model_id=card["model_id"],
        vocabulary=vocabulary,
        idf=idf,
        verses=verses,
        verse_vectors=verse_vectors,
    )

