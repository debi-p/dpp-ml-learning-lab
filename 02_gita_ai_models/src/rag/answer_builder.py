MODEL_ID = "dpp-gita-rag-assistant-v2"


def daily_life_sentence(question, primary_source):
    tags = (primary_source.get("tags") or "").lower()
    question_lower = question.lower()

    if "anger" in tags or "anger" in question_lower:
        return "In daily life, pause before reacting, notice the desire or expectation behind the anger, and bring the senses back under control."
    if "mind" in tags or "mind" in question_lower or "overthinking" in question_lower:
        return "In daily life, when the mind wanders, gently bring it back to the right action again and again."
    if "duty" in tags or "duty" in question_lower or "work" in question_lower:
        return "In daily life, focus on doing the right duty well, without becoming owned by the result."
    if "devotion" in tags or "devotion" in question_lower:
        return "In daily life, keep the mind connected to the Divine through steady remembrance and sincere action."
    return "In daily life, apply the teaching steadily in thought, action, and self-control."


def build_rag_answer(context):
    sources = context.get("sources", [])
    if not sources:
        return {
            "model_id": MODEL_ID,
            "answer": "I could not find a relevant Bhagavad Gita passage yet.",
            "sources": [],
        }

    primary = sources[0]
    secondary = sources[1:3]
    support_refs = ", ".join(source["reference"] for source in secondary)

    answer = (
        f"{primary['reference']} gives the main direction: {primary['translation']} "
        f"The practical message is that inner discipline matters more than reacting immediately. "
        f"{daily_life_sentence(context['question'], primary)}"
    )

    if support_refs:
        answer += f" Related support also appears in {support_refs}."

    return {
        "model_id": MODEL_ID,
        "question": context["question"],
        "answer": answer.strip(),
        "sources": [
            {
                "chapter": source["chapter"],
                "verse": source["verse"],
                "score": source["score"],
                "translation": source["translation"],
                "commentary": source["commentary"],
                "tags": source["tags"],
            }
            for source in sources
        ],
        "context": context,
    }

