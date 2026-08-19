from dataclasses import dataclass

from src.vectorize import build_idf, cosine_similarity, vectorize_tfidf
from src.vocabulary import build_vocabulary


@dataclass
class SearchModel:
    model_id: str
    vocabulary: dict
    idf: dict
    verses: list
    verse_vectors: list


@dataclass
class SearchResult:
    verse: object
    score: float
    matched_words: list


def build_search_model(verses, model_id="dpp-gita-search-assistant-v1", max_vocab_size=5000):
    texts = [verse.searchable_text() for verse in verses]
    vocabulary = build_vocabulary(texts, max_size=max_vocab_size)
    idf = build_idf(texts, vocabulary)
    verse_vectors = [vectorize_tfidf(text, vocabulary, idf) for text in texts]
    return SearchModel(model_id=model_id, vocabulary=vocabulary, idf=idf, verses=verses, verse_vectors=verse_vectors)


def search(model, question, top_k=3):
    question_vector = vectorize_tfidf(question, model.vocabulary, model.idf)
    results = []

    for verse, verse_vector in zip(model.verses, model.verse_vectors):
        score = cosine_similarity(question_vector, verse_vector)
        results.append(SearchResult(verse=verse, score=score, matched_words=[]))

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]
