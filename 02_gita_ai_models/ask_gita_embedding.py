import argparse

from src.answer_builder import build_answer
from src.embedding.search import search_embeddings
from src.embedding.storage import load_embedding_artifacts


def main():
    parser = argparse.ArgumentParser(description="Ask the neural embedding Gita assistant.")
    parser.add_argument("question", nargs="*", help="Question to ask.")
    parser.add_argument("--model-dir", default="models/dpp-gita-embedding-small-v1")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    question = " ".join(args.question).strip() or "How can I control anger?"
    index = load_embedding_artifacts(args.model_dir)
    results = search_embeddings(index, question, top_k=args.top_k)
    answer = build_answer(question, results)

    print(f"Model: {index.model_id}")
    print(f"Question: {question}")
    print()
    print(answer["answer"])
    print()
    print("Sources:")
    for source in answer["sources"]:
        print(f"- Chapter {source['chapter']}, Verse {source['verse']} | score={source['score']}")


if __name__ == "__main__":
    main()

