from collections import Counter

from src.text import tokenize


def build_vocabulary(texts, max_size=5000):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    tokens = ["<UNK>"]
    for token, _ in counts.most_common(max(0, max_size - 1)):
        if token not in tokens:
            tokens.append(token)

    return {token: index for index, token in enumerate(tokens)}

