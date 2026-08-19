from src.answer_builder import build_answer
from src.retrieval import search
from src.storage import load_search_model


class GitaAssistant:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_id, models_dir="models"):
        model = load_search_model(f"{models_dir}/{model_id}")
        return cls(model)

    def ask(self, question, top_k=3):
        results = search(self.model, question, top_k=top_k)
        response = build_answer(question, results)
        response["model_id"] = self.model.model_id
        response["question"] = question
        return response
