from dataclasses import dataclass

import numpy as np


def softmax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def causal_mask(context_length):
    return np.triu(np.ones((context_length, context_length), dtype=bool), k=1)


@dataclass
class TransformerForwardOutput:
    logits: np.ndarray
    probabilities: np.ndarray
    combined_embeddings: np.ndarray
    token_embeddings: np.ndarray
    position_embeddings: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    attention_scores: np.ndarray
    attention_weights: np.ndarray
    attention_output: np.ndarray
    attention_projected: np.ndarray
    hidden_linear: np.ndarray
    hidden_states: np.ndarray


@dataclass
class TinyTransformerModel:
    token_embedding: np.ndarray
    position_embedding: np.ndarray
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_attention_out: np.ndarray
    w_hidden: np.ndarray
    b_hidden: np.ndarray
    w_output: np.ndarray
    b_output: np.ndarray

    @classmethod
    def create(cls, vocab_size, context_length, d_model=64, hidden_size=128, seed=7):
        rng = np.random.default_rng(seed)
        scale = 0.02
        return cls(
            token_embedding=rng.normal(0.0, scale, size=(vocab_size, d_model)),
            position_embedding=rng.normal(0.0, scale, size=(context_length, d_model)),
            w_q=rng.normal(0.0, scale, size=(d_model, d_model)),
            w_k=rng.normal(0.0, scale, size=(d_model, d_model)),
            w_v=rng.normal(0.0, scale, size=(d_model, d_model)),
            w_attention_out=rng.normal(0.0, scale, size=(d_model, d_model)),
            w_hidden=rng.normal(0.0, scale, size=(d_model, hidden_size)),
            b_hidden=np.zeros(hidden_size),
            w_output=rng.normal(0.0, scale, size=(hidden_size, vocab_size)),
            b_output=np.zeros(vocab_size),
        )

    @property
    def context_length(self):
        return self.position_embedding.shape[0]

    @property
    def d_model(self):
        return self.token_embedding.shape[1]

    def forward(self, input_ids):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape batch_size x context_length")
        if input_ids.shape[1] != self.context_length:
            raise ValueError("input context length does not match model context_length")

        token_embeddings = self.token_embedding[input_ids]
        position_embeddings = self.position_embedding[np.newaxis, :, :]
        x = token_embeddings + position_embeddings

        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        attention_scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(self.d_model)
        attention_scores = attention_scores.copy()
        attention_scores[:, causal_mask(self.context_length)] = -1e9
        attention_weights = softmax(attention_scores)
        attention_output = attention_weights @ v
        attention_projected = attention_output @ self.w_attention_out

        hidden_linear = attention_projected @ self.w_hidden + self.b_hidden
        hidden_states = np.maximum(0.0, hidden_linear)
        logits = hidden_states @ self.w_output + self.b_output
        probabilities = softmax(logits)

        return TransformerForwardOutput(
            logits=logits,
            probabilities=probabilities,
            combined_embeddings=x,
            token_embeddings=token_embeddings,
            position_embeddings=np.broadcast_to(position_embeddings, token_embeddings.shape),
            q=q,
            k=k,
            v=v,
            attention_scores=attention_scores,
            attention_weights=attention_weights,
            attention_output=attention_output,
            attention_projected=attention_projected,
            hidden_linear=hidden_linear,
            hidden_states=hidden_states,
        )
