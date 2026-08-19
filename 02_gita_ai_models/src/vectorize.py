import math
from collections import Counter

from src.text import tokenize


def build_idf(texts, vocabulary):
    document_count = len(texts)
    document_frequency = {token: 0 for token in vocabulary}

    for text in texts:
        seen = set(tokenize(text))
        for token in seen:
            if token in document_frequency:
                document_frequency[token] += 1

    return {
        token: math.log((1 + document_count) / (1 + document_frequency[token])) + 1
        for token in vocabulary
    }


def vectorize_tfidf(text, vocabulary, idf):
    counts = Counter(tokenize(text))
    total = sum(counts.values()) or 1
    vector = {}

    for token, count in counts.items():
        if token in vocabulary:
            index = vocabulary[token]
            tf = count / total
            vector[index] = tf * idf.get(token, 1.0)

    return vector


def cosine_similarity(left, right):
    if not left or not right:
        return 0.0

    shared = set(left).intersection(right)
    dot = sum(left[index] * right[index] for index in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)

