import argparse

from sdk.gita_assistant import GitaAssistant


def main():
    parser = argparse.ArgumentParser(description="Ask the from-scratch Bhagavad Gita assistant.")
    parser.add_argument("question", nargs="*", help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-id", default="dpp-gita-search-assistant-v1")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = "How can I control anger?"

    assistant = GitaAssistant.load(args.model_id)
    response = assistant.ask(question, top_k=args.top_k)

    print(f"Model: {response['model_id']}")
    print(f"Question: {response['question']}")
    print()
    print(response["answer"])
    print()
    print("Sources:")
    for source in response["sources"]:
        print(f"- Chapter {source['chapter']}, Verse {source['verse']} | score={source['score']}")


if __name__ == "__main__":
    main()

