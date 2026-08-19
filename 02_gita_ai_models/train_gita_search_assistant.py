from pathlib import Path

from src.dataset import load_verses_csv
from src.retrieval import build_search_model
from src.storage import save_search_model


MODEL_ID = "dpp-gita-search-assistant-v1"
DATASET_PATH = Path("data/gita_verses.csv")
MODEL_DIR = Path("models") / MODEL_ID


def main():
    verses = load_verses_csv(DATASET_PATH)
    if not verses:
        raise SystemExit("No verses found. Run PDF extraction and dataset build first.")

    model = build_search_model(verses, model_id=MODEL_ID)
    save_search_model(model, MODEL_DIR)
    print(f"Saved {MODEL_ID}")
    print(f"Verse rows: {len(verses)}")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    print(f"Model folder: {MODEL_DIR}")


if __name__ == "__main__":
    main()

