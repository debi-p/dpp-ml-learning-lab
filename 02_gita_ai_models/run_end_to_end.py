from scripts.build_gita_dataset import build_dataset
from scripts.extract_pdf_text import extract_pdf_text
from scripts.validate_gita_dataset import validate_dataset
from src.dataset import load_verses_csv
from src.retrieval import build_search_model
from src.storage import save_search_model


PDF_PATH = "source_pdfs/bhagavad-gita-as-it-is.pdf"
RAW_TEXT_PATH = "data/extracted_raw_text.txt"
CLEAN_TEXT_PATH = "data/gita_clean_text.txt"
DATASET_PATH = "data/gita_verses.csv"
MODEL_ID = "dpp-gita-search-assistant-v1"
MODEL_DIR = f"models/{MODEL_ID}"


def main():
    extraction = extract_pdf_text(PDF_PATH, RAW_TEXT_PATH)
    print(f"Extracted PDF pages: {extraction['pages']}")

    verses = build_dataset(RAW_TEXT_PATH, DATASET_PATH, clean_text_path=CLEAN_TEXT_PATH)
    print(f"Built dataset rows: {len(verses)}")

    report = validate_dataset(DATASET_PATH)
    print(f"Validated chapters: {report['chapter_count']}")
    print(f"Missing translations: {report['missing_translation_count']}")

    model = build_search_model(load_verses_csv(DATASET_PATH), model_id=MODEL_ID)
    save_search_model(model, MODEL_DIR)
    print(f"Saved model: {MODEL_DIR}")
    print(f"Vocabulary size: {len(model.vocabulary)}")


if __name__ == "__main__":
    main()

