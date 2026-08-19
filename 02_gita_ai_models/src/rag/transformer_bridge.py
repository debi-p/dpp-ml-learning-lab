MODEL_ID = "dpp-gita-rag-transformer-v1"


def build_transformer_prompt(context, max_source_chars=900):
    question = context["question"]
    source_lines = []
    for source in context.get("sources", []):
        text = " ".join([source.get("translation", ""), source.get("commentary", "")]).strip()
        if len(text) > max_source_chars:
            text = text[:max_source_chars].rsplit(" ", 1)[0].strip() + "..."
        source_lines.append(f"{source['reference']}: {text}")

    sources_text = "\n".join(source_lines)
    return (
        "Use the Bhagavad Gita context to answer simply.\n"
        f"Question: {question}\n"
        "Context:\n"
        f"{sources_text}\n"
        "Answer:"
    )


def build_rag_transformer_answer(context, generated_text, generation_steps):
    answer = _extract_answer_text(generated_text)
    return {
        "model_id": MODEL_ID,
        "question": context["question"],
        "answer": answer,
        "sources": [
            {
                "chapter": source["chapter"],
                "verse": source["verse"],
                "score": source["score"],
                "reference": source["reference"],
            }
            for source in context.get("sources", [])
        ],
        "generation_steps": generation_steps,
        "note": "Experimental RAG + tiny transformer path. Retrieval is reliable; generation is from a very small from-scratch transformer.",
    }


def _extract_answer_text(generated_text):
    marker = "answer"
    lowered = generated_text.lower()
    index = lowered.rfind(marker)
    if index >= 0:
        extracted = generated_text[index + len(marker) :].strip(" :\n\t")
        if extracted:
            return extracted
    return generated_text.strip()
