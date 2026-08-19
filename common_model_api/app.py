import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "model_registry" / "registry.json"

app = FastAPI(title="DPP Common Model API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    model_id: str
    input: str


class InspectTrainingStepRequest(PredictRequest):
    correct_label: str
    learning_rate: float = 0.1


def load_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def registry_models():
    return load_registry().get("models", [])


def find_model_entry(model_id):
    for model in registry_models():
        if model.get("model_id") == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")


def clear_local_modules(*roots):
    for name in list(sys.modules):
        if name in roots or any(name.startswith(f"{root}.") for root in roots):
            del sys.modules[name]


@lru_cache(maxsize=16)
def load_model(model_id):
    entry = find_model_entry(model_id)

    if entry["type"] == "email_classifier_from_scratch":
        return load_email_classifier(entry)

    if entry["type"] == "gita_rag_assistant_from_scratch":
        return load_gita_rag_assistant(entry)

    if entry["type"] == "gita_tiny_transformer_from_scratch":
        return load_gita_tiny_transformer(entry)

    if entry["type"] == "gita_rag_transformer_from_scratch":
        return load_gita_rag_transformer(entry)

    raise HTTPException(status_code=400, detail=f"Unsupported model type: {entry['type']}")


def load_email_classifier(entry):
    clear_local_modules("src", "sdk")
    phase_dir = (ROOT_DIR / "01_foundational_neural_network").resolve()
    if str(phase_dir) not in sys.path:
        sys.path.insert(0, str(phase_dir))

    classifier_module = importlib.import_module("sdk.classifier")
    model_path = (ROOT_DIR / "model_registry" / entry["path"]).resolve()
    return classifier_module.EmailClassifier.load(entry["model_id"], models_dir=model_path.parent)


class GitaRagAssistantAdapter:
    def __init__(self, entry, index, search_embeddings, build_augmented_context, build_rag_answer):
        self.entry = entry
        self.index = index
        self.search_embeddings = search_embeddings
        self.build_augmented_context = build_augmented_context
        self.build_rag_answer = build_rag_answer

    def metadata(self):
        return {
            "model_id": self.entry["model_id"],
            "target_type": "rag_answer",
            "retriever_model_id": self.entry["retriever_model_id"],
            "supports": self.entry.get("supports", []),
        }

    def predict(self, text):
        results = self.search_embeddings(self.index, text, top_k=3)
        context = self.build_augmented_context(text, results, max_sources=3)
        answer = self.build_rag_answer(context)
        return {
            "model_id": answer["model_id"],
            "retriever_model_id": self.index.model_id,
            "question": answer["question"],
            "answer": answer["answer"],
            "sources": answer["sources"],
        }

    def inspect_rag(self, text):
        results = self.search_embeddings(self.index, text, top_k=5)
        context = self.build_augmented_context(text, results, max_sources=5)
        answer = self.build_rag_answer(context)
        return {
            "model_id": self.entry["model_id"],
            "retriever_model_id": self.index.model_id,
            "input": {"text": text},
            "retrieval": {
                "algorithm": "neural embedding cosine similarity",
                "embedding_shape": [int(self.index.verse_embeddings.shape[0]), int(self.index.verse_embeddings.shape[1])],
                "results": [
                    {
                        "chapter": result.verse.chapter,
                        "verse": result.verse.verse,
                        "score": round(result.score, 6),
                        "translation": result.verse.translation,
                        "tags": result.verse.tags,
                    }
                    for result in results
                ],
            },
            "augmented_context": context,
            "answer": answer,
        }


def load_gita_rag_assistant(entry):
    clear_local_modules("src", "sdk")
    phase_dir = (ROOT_DIR / "02_gita_ai_models").resolve()
    if str(phase_dir) not in sys.path:
        sys.path.insert(0, str(phase_dir))

    storage_module = importlib.import_module("src.embedding.storage")
    search_module = importlib.import_module("src.embedding.search")
    context_module = importlib.import_module("src.rag.context_builder")
    answer_module = importlib.import_module("src.rag.answer_builder")
    model_path = (ROOT_DIR / "model_registry" / entry["path"]).resolve()
    index = storage_module.load_embedding_artifacts(model_path)
    return GitaRagAssistantAdapter(
        entry=entry,
        index=index,
        search_embeddings=search_module.search_embeddings,
        build_augmented_context=context_module.build_augmented_context,
        build_rag_answer=answer_module.build_rag_answer,
    )


class GitaTinyTransformerAdapter:
    def __init__(self, entry, model, vocabulary, config, card, generate_text, default_avoid_tokens):
        self.entry = entry
        self.model = model
        self.vocabulary = vocabulary
        self.config = config
        self.card = card
        self.generate_text = generate_text
        self.avoid_tokens = default_avoid_tokens

    def metadata(self):
        return {
            "model_id": self.entry["model_id"],
            "target_type": "text_generation",
            "supports": self.entry.get("supports", []),
            "context_length": self.config.get("context_length"),
            "vocabulary_size": len(self.vocabulary),
            "pretrained_model_used": self.card.get("pretrained_model_used", None),
        }

    def predict(self, text):
        result = self.generate_text(
            self.model,
            self.vocabulary,
            prompt=text,
            max_new_tokens=20,
            temperature=0.8,
            top_k=5,
            avoid_tokens=self.avoid_tokens,
        )
        return {
            "model_id": self.entry["model_id"],
            "prompt": text,
            "generated_text": result.text,
            "generated_tokens": result.generated_tokens,
            "context_length": self.config.get("context_length"),
            "vocabulary_size": len(self.vocabulary),
            "generation_config": {"max_new_tokens": 20, "temperature": 0.8, "top_k": 5, "avoid_common_tokens": True},
            "note": "Standalone tiny transformer generation. This is not yet integrated with RAG retrieval.",
        }

    def inspect_transformer(self, text):
        result = self.generate_text(
            self.model,
            self.vocabulary,
            prompt=text,
            max_new_tokens=10,
            temperature=0.8,
            top_k=5,
            avoid_tokens=self.avoid_tokens,
        )
        return {
            "model_id": self.entry["model_id"],
            "input": {"prompt": text},
            "tokenization": {
                "prompt_tokens": result.prompt_tokens,
                "token_ids": result.token_ids[: len(result.prompt_tokens)],
                "context_length": self.config.get("context_length"),
                "vocabulary_size": len(self.vocabulary),
            },
            "generation_steps": result.steps,
            "generated_text": result.text,
            "model_config": self.config,
            "generation_config": {"max_new_tokens": 10, "temperature": 0.8, "top_k": 5, "avoid_common_tokens": True},
            "model_card": self.card,
            "rag_integration": {
                "enabled": False,
                "note": "RAG assistant and tiny transformer are separate models right now.",
            },
        }


def load_gita_tiny_transformer(entry):
    clear_local_modules("src", "sdk")
    phase_dir = (ROOT_DIR / "02_gita_ai_models").resolve()
    if str(phase_dir) not in sys.path:
        sys.path.insert(0, str(phase_dir))

    storage_module = importlib.import_module("src.transformer.storage")
    generate_module = importlib.import_module("src.transformer.generate")
    model_path = (ROOT_DIR / "model_registry" / entry["path"]).resolve()
    model, vocabulary, config, card = storage_module.load_transformer_artifacts(model_path)
    return GitaTinyTransformerAdapter(
        entry=entry,
        model=model,
        vocabulary=vocabulary,
        config=config,
        card=card,
        generate_text=generate_module.generate_text,
        default_avoid_tokens=generate_module.DEFAULT_AVOID_TOKENS,
    )


class GitaRagTransformerAdapter:
    def __init__(
        self,
        entry,
        index,
        transformer,
        vocabulary,
        transformer_config,
        transformer_card,
        search_embeddings,
        build_augmented_context,
        build_transformer_prompt,
        build_rag_transformer_answer,
        generate_text,
        default_avoid_tokens,
    ):
        self.entry = entry
        self.index = index
        self.transformer = transformer
        self.vocabulary = vocabulary
        self.transformer_config = transformer_config
        self.transformer_card = transformer_card
        self.search_embeddings = search_embeddings
        self.build_augmented_context = build_augmented_context
        self.build_transformer_prompt = build_transformer_prompt
        self.build_rag_transformer_answer = build_rag_transformer_answer
        self.generate_text = generate_text
        self.avoid_tokens = default_avoid_tokens

    def metadata(self):
        return {
            "model_id": self.entry["model_id"],
            "target_type": "rag_transformer_answer",
            "retriever_model_id": self.entry["retriever_model_id"],
            "generator_model_id": self.entry["generator_model_id"],
            "supports": self.entry.get("supports", []),
            "context_length": self.transformer_config.get("context_length"),
            "vocabulary_size": len(self.vocabulary),
            "pretrained_model_used": False,
        }

    def predict(self, text):
        trace = self._run(text, top_k=3, max_new_tokens=30)
        return {
            "model_id": self.entry["model_id"],
            "retriever_model_id": self.index.model_id,
            "generator_model_id": self.transformer_card["model_id"],
            "question": trace["question"],
            "answer": trace["answer"],
            "sources": trace["sources"],
            "generation_steps": trace["generation_steps"],
            "note": trace["note"],
        }

    def inspect_rag_transformer(self, text):
        trace = self._run(text, top_k=5, max_new_tokens=20)
        return trace

    def _run(self, text, top_k, max_new_tokens):
        results = self.search_embeddings(self.index, text, top_k=top_k)
        context = self.build_augmented_context(text, results, max_sources=top_k)
        prompt = self.build_transformer_prompt(context)
        generation = self.generate_text(
            self.transformer,
            self.vocabulary,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=5,
            avoid_tokens=self.avoid_tokens,
        )
        generated_answer_text = " ".join(generation.generated_tokens)
        answer = self.build_rag_transformer_answer(context, generated_answer_text, generation.steps)
        return {
            **answer,
            "retriever_model_id": self.index.model_id,
            "generator_model_id": self.transformer_card["model_id"],
            "input": {"question": text},
            "retrieval": {
                "algorithm": "neural embedding cosine similarity",
                "results": [
                    {
                        "chapter": result.verse.chapter,
                        "verse": result.verse.verse,
                        "score": round(result.score, 6),
                        "translation": result.verse.translation,
                        "tags": result.verse.tags,
                    }
                    for result in results
                ],
            },
            "augmented_context": context,
            "transformer_prompt": prompt,
            "generated_text": generation.text,
            "transformer_config": self.transformer_config,
            "generation_config": {"max_new_tokens": max_new_tokens, "temperature": 0.8, "top_k": 5, "avoid_common_tokens": True},
        }


def load_gita_rag_transformer(entry):
    clear_local_modules("src", "sdk")
    phase_dir = (ROOT_DIR / "02_gita_ai_models").resolve()
    if str(phase_dir) not in sys.path:
        sys.path.insert(0, str(phase_dir))

    embedding_storage_module = importlib.import_module("src.embedding.storage")
    search_module = importlib.import_module("src.embedding.search")
    transformer_storage_module = importlib.import_module("src.transformer.storage")
    generate_module = importlib.import_module("src.transformer.generate")
    context_module = importlib.import_module("src.rag.context_builder")
    bridge_module = importlib.import_module("src.rag.transformer_bridge")

    embedding_path = (ROOT_DIR / "model_registry" / entry["embedding_model_path"]).resolve()
    transformer_path = (ROOT_DIR / "model_registry" / entry["transformer_model_path"]).resolve()
    index = embedding_storage_module.load_embedding_artifacts(embedding_path)
    transformer, vocabulary, transformer_config, transformer_card = transformer_storage_module.load_transformer_artifacts(
        transformer_path
    )
    return GitaRagTransformerAdapter(
        entry=entry,
        index=index,
        transformer=transformer,
        vocabulary=vocabulary,
        transformer_config=transformer_config,
        transformer_card=transformer_card,
        search_embeddings=search_module.search_embeddings,
        build_augmented_context=context_module.build_augmented_context,
        build_transformer_prompt=bridge_module.build_transformer_prompt,
        build_rag_transformer_answer=bridge_module.build_rag_transformer_answer,
        generate_text=generate_module.generate_text,
        default_avoid_tokens=generate_module.DEFAULT_AVOID_TOKENS,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    models = []
    for entry in registry_models():
        model_info = dict(entry)
        try:
            model_info.update(load_model(entry["model_id"]).metadata())
        except HTTPException:
            model_info.setdefault("supports", entry.get("supports", []))
        models.append(model_info)
    return models


@app.post("/predict")
def predict(request: PredictRequest):
    model = load_model(request.model_id)
    return model.predict(request.input)


@app.post("/inspect-forward")
def inspect_forward(request: PredictRequest):
    classifier = load_model(request.model_id)
    if not hasattr(classifier, "inspect_forward"):
        raise HTTPException(status_code=400, detail=f"Model does not support inspect_forward: {request.model_id}")
    return classifier.inspect_forward(request.input)


@app.post("/inspect-training-step")
def inspect_training_step(request: InspectTrainingStepRequest):
    classifier = load_model(request.model_id)
    if not hasattr(classifier, "inspect_training_step"):
        raise HTTPException(status_code=400, detail=f"Model does not support inspect_training_step: {request.model_id}")
    try:
        return classifier.inspect_training_step(
            request.input,
            correct_label=request.correct_label,
            learning_rate=request.learning_rate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/inspect-rag")
def inspect_rag(request: PredictRequest):
    model = load_model(request.model_id)
    if not hasattr(model, "inspect_rag"):
        raise HTTPException(status_code=400, detail=f"Model does not support inspect_rag: {request.model_id}")
    return model.inspect_rag(request.input)


@app.post("/inspect-transformer")
def inspect_transformer(request: PredictRequest):
    model = load_model(request.model_id)
    if not hasattr(model, "inspect_transformer"):
        raise HTTPException(status_code=400, detail=f"Model does not support inspect_transformer: {request.model_id}")
    return model.inspect_transformer(request.input)


@app.post("/inspect-rag-transformer")
def inspect_rag_transformer(request: PredictRequest):
    model = load_model(request.model_id)
    if not hasattr(model, "inspect_rag_transformer"):
        raise HTTPException(status_code=400, detail=f"Model does not support inspect_rag_transformer: {request.model_id}")
    return model.inspect_rag_transformer(request.input)
