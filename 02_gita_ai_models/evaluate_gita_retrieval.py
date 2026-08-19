import argparse
import csv
from pathlib import Path

from src.dataset import load_verses_csv
from src.embedding.search import search_embeddings
from src.embedding.storage import load_embedding_artifacts
from src.retrieval import build_search_model, search


def load_eval_rows(path, limit=None):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        return rows[:limit]
    return rows


def score_results(rows, predictor, top_k_values=(1, 3, 5)):
    correct = {k: 0 for k in top_k_values}
    for row in rows:
        expected = (row["positive_chapter"].strip(), row["positive_verse"].strip())
        results = predictor(row["question"], max(top_k_values))
        result_keys = [(result.verse.chapter, result.verse.verse) for result in results]
        for k in top_k_values:
            if expected in result_keys[:k]:
                correct[k] += 1
    return {f"top_{k}_accuracy": correct[k] / max(1, len(rows)) for k in top_k_values}


def main():
    parser = argparse.ArgumentParser(description="Compare TF-IDF and neural embedding retrieval.")
    parser.add_argument("--pairs", default="data/gita_question_pairs.csv")
    parser.add_argument("--verses", default="data/gita_verses.csv")
    parser.add_argument("--embedding-model-dir", default="models/dpp-gita-embedding-small-v1")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    rows = load_eval_rows(args.pairs, limit=args.limit)
    verses = load_verses_csv(args.verses)
    tfidf_model = build_search_model(verses)
    embedding_index = load_embedding_artifacts(args.embedding_model_dir)

    tfidf_scores = score_results(rows, lambda question, top_k: search(tfidf_model, question, top_k=top_k))
    embedding_scores = score_results(rows, lambda question, top_k: search_embeddings(embedding_index, question, top_k=top_k))

    print("Rows evaluated:", len(rows))
    print("TF-IDF baseline:")
    for key, value in tfidf_scores.items():
        print(f"  {key}: {value:.3f}")
    print("Neural embedding:")
    for key, value in embedding_scores.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
