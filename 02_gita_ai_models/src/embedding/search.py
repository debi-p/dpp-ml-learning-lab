from dataclasses import dataclass

from src.embedding.model import cosine_scores


@dataclass
class EmbeddingSearchIndex:
    model_id: str
    model: object
    vocabulary: dict
    verses: list
    verse_embeddings: object
    max_length: int


@dataclass
class EmbeddingSearchResult:
    verse: object
    score: float
    matched_words: list


def search_embeddings(index, question, top_k=3, query_vector=None):
    if query_vector is None:
        query_vector = index.model.embed_text(question, index.vocabulary, max_length=index.max_length)
    scores = cosine_scores(query_vector, index.verse_embeddings)
    order = scores.argsort()[::-1][:top_k]
    return [EmbeddingSearchResult(verse=index.verses[int(i)], score=float(scores[int(i)]), matched_words=[]) for i in order]
