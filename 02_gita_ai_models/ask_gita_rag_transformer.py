import argparse

from src.embedding.search import search_embeddings
from src.embedding.storage import load_embedding_artifacts
from src.rag.context_builder import build_augmented_context
from src.rag.transformer_bridge import build_rag_transformer_answer, build_transformer_prompt
from src.transformer.generate import DEFAULT_AVOID_TOKENS, generate_text
from src.transformer.storage import load_transformer_artifacts


MODEL_ID = "dpp-gita-rag-transformer-v1"


def ask_rag_transformer(
    question,
    embedding_model_dir="models/dpp-gita-embedding-small-v1",
    transformer_model_dir="models/dpp-gita-tiny-transformer-v1",
    top_k=3,
    max_new_tokens=30,
    temperature=0.8,
    generation_top_k=5,
    avoid_common_tokens=True,
):
    index = load_embedding_artifacts(embedding_model_dir)
    transformer, vocabulary, config, card = load_transformer_artifacts(transformer_model_dir)
    results = search_embeddings(index, question, top_k=top_k)
    context = build_augmented_context(question, results, max_sources=top_k)
    prompt = build_transformer_prompt(context)
    generation = generate_text(
        transformer,
        vocabulary,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=generation_top_k,
        avoid_tokens=DEFAULT_AVOID_TOKENS if avoid_common_tokens else None,
    )
    generated_answer_text = " ".join(generation.generated_tokens)
    response = build_rag_transformer_answer(context, generated_answer_text, generation.steps)
    return {
        **response,
        "retriever_model_id": index.model_id,
        "generator_model_id": card["model_id"],
        "transformer_prompt": prompt,
        "generated_text": generation.text,
        "transformer_config": config,
    }


def main():
    parser = argparse.ArgumentParser(description="Ask the experimental RAG + tiny transformer Gita assistant.")
    parser.add_argument("question", nargs="*", help="Question to ask.")
    parser.add_argument("--embedding-model-dir", default="models/dpp-gita-embedding-small-v1")
    parser.add_argument("--transformer-model-dir", default="models/dpp-gita-tiny-transformer-v1")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--generation-top-k", type=int, default=5)
    parser.add_argument("--allow-common-tokens", action="store_true")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--show-steps", action="store_true")
    args = parser.parse_args()

    question = " ".join(args.question).strip() or "How can I control anger?"
    response = ask_rag_transformer(
        question=question,
        embedding_model_dir=args.embedding_model_dir,
        transformer_model_dir=args.transformer_model_dir,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        generation_top_k=args.generation_top_k,
        avoid_common_tokens=not args.allow_common_tokens,
    )

    print(f"Model: {response['model_id']}")
    print(f"Retriever: {response['retriever_model_id']}")
    print(f"Generator: {response['generator_model_id']}")
    print(f"Question: {response['question']}")
    print()
    print(response["answer"])
    print()
    print("Sources:")
    for source in response["sources"]:
        print(f"- Chapter {source['chapter']}, Verse {source['verse']} | score={source['score']}")

    if args.show_prompt:
        print()
        print("Transformer prompt:")
        print(response["transformer_prompt"])

    if args.show_steps:
        print()
        print("Generation steps:")
        for index, step in enumerate(response["generation_steps"], start=1):
            top = ", ".join(f"{item['token']}={item['probability']:.3f}" for item in step["top_tokens"][:5])
            print(f"{index}. {step['next_token']} | {top}")


if __name__ == "__main__":
    main()
