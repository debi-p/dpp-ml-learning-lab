import argparse

from src.embedding.search import search_embeddings
from src.embedding.storage import load_embedding_artifacts
from src.rag.answer_builder import build_rag_answer
from src.rag.context_builder import build_augmented_context


def main():
    parser = argparse.ArgumentParser(description="Ask the clean RAG Bhagavad Gita assistant.")
    parser.add_argument("question", nargs="*", help="Question to ask.")
    parser.add_argument("--embedding-model-dir", default="models/dpp-gita-embedding-small-v1")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    question = " ".join(args.question).strip() or "How can I control anger?"
    index = load_embedding_artifacts(args.embedding_model_dir)
    results = search_embeddings(index, question, top_k=args.top_k)
    context = build_augmented_context(question, results, max_sources=args.top_k)
    response = build_rag_answer(context)

    print(f"Model: {response['model_id']}")
    print(f"Retriever: {index.model_id}")
    print(f"Question: {response['question']}")
    print()
    print(response["answer"])
    print()
    print("Sources:")
    for source in response["sources"]:
        print(f"- Chapter {source['chapter']}, Verse {source['verse']} | score={source['score']}")


if __name__ == "__main__":
    main()
