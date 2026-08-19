from src.text import normalize_for_display


def build_answer(question, results):
    if not question or not question.strip():
        raise ValueError("Question is required.")

    useful_results = [result for result in results if result.score > 0]
    selected = useful_results or list(results[:1])

    if not selected:
        return {
            "answer": "I could not find a relevant Bhagavad Gita passage yet.",
            "sources": [],
        }

    primary = selected[0].verse
    translation = normalize_for_display(primary.translation)
    commentary = normalize_for_display(primary.commentary)
    answer = (
        f"Based on Chapter {primary.chapter}, Verse {primary.verse}, "
        f"the Gita points to this idea: {translation} "
    )

    if commentary:
        answer += f"In simple daily life, this means: {commentary}"

    sources = [
        {
            "chapter": result.verse.chapter,
            "verse": result.verse.verse,
            "score": round(result.score, 6),
            "translation": normalize_for_display(result.verse.translation),
            "commentary": normalize_for_display(result.verse.commentary),
            "tags": result.verse.tags,
            "matched_words": result.matched_words,
        }
        for result in selected
    ]

    return {"answer": answer.strip(), "sources": sources}
