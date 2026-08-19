from collections import Counter

from src.text import tokenize


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_transformer_vocabulary(texts, max_size=5000):
    if max_size < 2:
        raise ValueError("max_size must leave room for <PAD> and <UNK>")

    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if token in vocabulary:
            continue
        if len(vocabulary) >= max_size:
            break
        vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_text(text, vocabulary, max_length):
    if max_length < 1:
        raise ValueError("max_length must be positive")

    pad_id = vocabulary[PAD_TOKEN]
    unk_id = vocabulary[UNK_TOKEN]
    token_ids = [vocabulary.get(token, unk_id) for token in tokenize(text)]
    token_ids = token_ids[:max_length]

    if len(token_ids) < max_length:
        token_ids.extend([pad_id] * (max_length - len(token_ids)))

    return token_ids


def decode_token_ids(token_ids, vocabulary):
    reverse = {token_id: token for token, token_id in vocabulary.items()}
    tokens = []
    for token_id in token_ids:
        token = reverse.get(token_id, UNK_TOKEN)
        if token == PAD_TOKEN:
            continue
        tokens.append(token)
    return " ".join(tokens)
