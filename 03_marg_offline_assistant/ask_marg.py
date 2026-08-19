import argparse

from backend.intent_model import IntentModel
from backend.paths import intent_model_dir, model_dir
from backend.rag_engine import MargRagEngine


def main():
    parser = argparse.ArgumentParser(description="Ask Marg offline from the terminal.")
    parser.add_argument("question", nargs="*", help="Question to ask Marg.")
    args = parser.parse_args()

    question = " ".join(args.question).strip() or "How can I control anger?"
    engine = MargRagEngine.load(model_dir())
    intent_model = IntentModel.load(intent_model_dir())
    response = engine.ask(question, intent_model=intent_model)

    print("Marg")
    print(f"Question: {response['question']}")
    print(f"Intent: {response.get('intent')} ({response.get('intent_confidence'):.3f})")
    print()
    print(response["answer"])
    print()
    print("Sources:")
    for source in response["sources"]:
        print(f"- {source['reference']} | score={source['score']}")


if __name__ == "__main__":
    main()
