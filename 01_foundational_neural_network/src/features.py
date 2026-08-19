from collections import Counter

import numpy as np


def build_vocabulary(tokenized_messages, max_size=1000):
    counts = Counter()
    for tokens in tokenized_messages:
        counts.update(tokens)

    most_common = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_size]
    return {word: index for index, (word, _) in enumerate(most_common)}


def vectorize_tokens(tokens, vocabulary):
    vector = np.zeros(len(vocabulary), dtype=float)
    for token in tokens:
        index = vocabulary.get(token)
        if index is not None:
            vector[index] += 1.0
    return vector
