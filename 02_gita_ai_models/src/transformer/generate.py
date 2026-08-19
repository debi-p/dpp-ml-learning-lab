from dataclasses import dataclass

import numpy as np

from src.text import tokenize
from src.transformer.tokenizer import PAD_TOKEN, UNK_TOKEN, decode_token_ids


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: list
    generated_tokens: list
    token_ids: list
    steps: list


DEFAULT_AVOID_TOKENS = {
    "the",
    "of",
    "is",
    "and",
    "to",
    "a",
    "in",
    "he",
    "his",
    "are",
    "that",
    "this",
    "it",
    "as",
    "for",
    "with",
}


def generate_text(
    model,
    vocabulary,
    prompt,
    max_new_tokens=20,
    temperature=0.0,
    top_k=5,
    seed=7,
    avoid_tokens=None,
):
    rng = np.random.default_rng(seed)
    reverse_vocabulary = {token_id: token for token, token_id in vocabulary.items()}
    pad_id = vocabulary[PAD_TOKEN]
    unk_id = vocabulary[UNK_TOKEN]
    token_ids = [vocabulary.get(token, unk_id) for token in tokenize(prompt)]
    prompt_tokens = [reverse_vocabulary.get(token_id, UNK_TOKEN) for token_id in token_ids]
    steps = []

    for _ in range(max_new_tokens):
        context = token_ids[-model.context_length :]
        if len(context) < model.context_length:
            context = [pad_id] * (model.context_length - len(context)) + context

        output = model.forward(np.array([context], dtype=np.int64))
        probabilities = output.probabilities[0, -1].copy()
        probabilities[pad_id] = 0.0
        probabilities[unk_id] = 0.0
        for token in avoid_tokens or set():
            token_id = vocabulary.get(token)
            if token_id is not None:
                probabilities[token_id] = 0.0
        probabilities = probabilities / np.sum(probabilities)
        candidate_probabilities = _apply_top_k(probabilities, top_k=top_k)
        next_id = _choose_next_token(candidate_probabilities, temperature=temperature, rng=rng)
        next_token = reverse_vocabulary.get(next_id, UNK_TOKEN)
        token_ids.append(next_id)

        top_indices = np.argsort(candidate_probabilities)[::-1][:top_k]
        steps.append(
            {
                "context_ids": context,
                "context_text": decode_token_ids(context, vocabulary),
                "next_token_id": int(next_id),
                "next_token": next_token,
                "top_tokens": [
                    {
                        "token": reverse_vocabulary.get(int(index), UNK_TOKEN),
                        "probability": float(candidate_probabilities[index]),
                    }
                    for index in top_indices
                ],
            }
        )

    generated_ids = token_ids[len(prompt_tokens) :]
    generated_tokens = [reverse_vocabulary.get(token_id, UNK_TOKEN) for token_id in generated_ids]
    return GenerationResult(
        text=decode_token_ids(token_ids, vocabulary),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        token_ids=token_ids,
        steps=steps,
    )


def _choose_next_token(probabilities, temperature, rng):
    if temperature <= 0.0:
        return int(np.argmax(probabilities))

    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    adjusted = np.exp(logits - np.max(logits))
    adjusted = adjusted / np.sum(adjusted)
    return int(rng.choice(len(adjusted), p=adjusted))


def _apply_top_k(probabilities, top_k):
    if top_k is None or top_k <= 0 or top_k >= len(probabilities):
        return probabilities

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    filtered = np.zeros_like(probabilities)
    filtered[top_indices] = probabilities[top_indices]
    total = np.sum(filtered)
    if total <= 0:
        return probabilities
    return filtered / total
