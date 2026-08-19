import json
from collections import Counter
from pathlib import Path

from src.text import tokenize


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_embedding_vocabulary(texts, max_size=8000):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    tokens = [PAD_TOKEN, UNK_TOKEN]
    for token, _ in counts.most_common(max(0, max_size - len(tokens))):
        if token not in tokens:
            tokens.append(token)

    return {token: index for index, token in enumerate(tokens)}


def encode_text(text, vocabulary, max_length=32):
    unk_id = vocabulary[UNK_TOKEN]
    pad_id = vocabulary[PAD_TOKEN]
    ids = [vocabulary.get(token, unk_id) for token in tokenize(text)]
    ids = ids[:max_length]
    if len(ids) < max_length:
        ids.extend([pad_id] * (max_length - len(ids)))
    return ids


def save_vocabulary(vocabulary, path):
    Path(path).write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")


def load_vocabulary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

