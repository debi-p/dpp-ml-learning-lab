from dataclasses import dataclass

from src.text import tokenize
from src.transformer.tokenizer import PAD_TOKEN, UNK_TOKEN


@dataclass(frozen=True)
class NextTokenExample:
    input_ids: list
    target_id: int


def build_next_token_examples(texts, vocabulary, context_length):
    if context_length < 1:
        raise ValueError("context_length must be positive")

    pad_id = vocabulary[PAD_TOKEN]
    unk_id = vocabulary[UNK_TOKEN]
    examples = []

    for text in texts:
        token_ids = [vocabulary.get(token, unk_id) for token in tokenize(text)]
        for position in range(1, len(token_ids)):
            start = max(0, position - context_length)
            context = token_ids[start:position]
            if len(context) < context_length:
                context = [pad_id] * (context_length - len(context)) + context
            examples.append(NextTokenExample(input_ids=context, target_id=token_ids[position]))

    return examples
