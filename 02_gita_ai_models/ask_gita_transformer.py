import argparse

from src.transformer.generate import DEFAULT_AVOID_TOKENS, generate_text
from src.transformer.storage import load_transformer_artifacts


MODEL_ID = "dpp-gita-tiny-transformer-v1"


def ask_transformer(
    prompt,
    model_dir=f"models/{MODEL_ID}",
    max_new_tokens=20,
    temperature=0.0,
    top_k=5,
    avoid_common_tokens=False,
):
    model, vocabulary, config, card = load_transformer_artifacts(model_dir)
    result = generate_text(
        model,
        vocabulary,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        avoid_tokens=DEFAULT_AVOID_TOKENS if avoid_common_tokens else None,
    )
    return card, config, result


def main():
    parser = argparse.ArgumentParser(description="Generate text with dpp-gita-tiny-transformer-v1.")
    parser.add_argument("prompt")
    parser.add_argument("--model-dir", default=f"models/{MODEL_ID}")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--avoid-common-tokens", action="store_true")
    parser.add_argument("--show-steps", action="store_true")
    args = parser.parse_args()

    card, config, result = ask_transformer(
        prompt=args.prompt,
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        avoid_common_tokens=args.avoid_common_tokens,
    )

    print(f"Model: {card['model_id']}")
    print(f"Context length: {config['context_length']}")
    print(f"Prompt: {args.prompt}")
    print()
    print(result.text)

    if args.show_steps:
        print()
        print("Generation steps:")
        for index, step in enumerate(result.steps, start=1):
            top = ", ".join(f"{item['token']}={item['probability']:.3f}" for item in step["top_tokens"][:5])
            print(f"{index}. context='{step['context_text']}' -> {step['next_token']} | {top}")


if __name__ == "__main__":
    main()
