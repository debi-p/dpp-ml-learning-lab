from src.text import normalize_for_display


def truncate_text(text, max_chars=260):
    cleaned = normalize_for_display(text)
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{cut}..."


def build_augmented_context(question, results, max_sources=3):
    if not question or not question.strip():
        raise ValueError("Question is required.")

    selected = [result for result in results if result.score > 0][:max_sources]
    if not selected:
        selected = list(results[:1])

    sources = []
    for result in selected:
        verse = result.verse
        sources.append(
            {
                "chapter": verse.chapter,
                "verse": verse.verse,
                "reference": f"Chapter {verse.chapter}, Verse {verse.verse}",
                "score": round(result.score, 6),
                "translation": truncate_text(verse.translation, max_chars=260),
                "commentary": truncate_text(verse.commentary, max_chars=320),
                "tags": verse.tags,
            }
        )

    return {
        "question": question.strip(),
        "sources": sources,
        "instruction": "Answer simply using only the retrieved Bhagavad Gita context.",
    }

