from dataclasses import dataclass

import numpy as np

from src.embedding.vocabulary import PAD_TOKEN, encode_text


EPSILON = 1e-8


def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < EPSILON:
        return vector
    return vector / norm


def cosine_scores(query_vector, matrix):
    query = l2_normalize(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norms = np.maximum(matrix_norms, EPSILON)
    normalized = matrix / matrix_norms
    return normalized @ query


@dataclass
class EmbeddingModel:
    token_embeddings: np.ndarray
    projection: np.ndarray
    pad_id: int = 0

    @classmethod
    def create(cls, vocab_size, token_dim=32, output_dim=64, seed=42):
        rng = np.random.default_rng(seed)
        token_embeddings = rng.normal(0.0, 0.05, size=(vocab_size, token_dim)).astype(np.float32)
        token_embeddings[0] = 0.0
        projection = rng.normal(0.0, 0.05, size=(token_dim, output_dim)).astype(np.float32)
        return cls(token_embeddings=token_embeddings, projection=projection, pad_id=0)

    def embed_token_ids(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        mask = ids != self.pad_id
        if not np.any(mask):
            pooled = np.zeros((self.token_embeddings.shape[1],), dtype=np.float32)
        else:
            pooled = self.token_embeddings[ids[mask]].mean(axis=0)
        projected = pooled @ self.projection
        return l2_normalize(projected).astype(np.float32)

    def embed_text(self, text, vocabulary, max_length=32):
        return self.embed_token_ids(encode_text(text, vocabulary, max_length=max_length))

    def train_triplet_step(self, question_ids, positive_ids, negative_ids, learning_rate=0.05, margin=0.25):
        cache = {}
        question = self._forward_with_cache(question_ids, cache, "q")
        positive = self._forward_with_cache(positive_ids, cache, "p")
        negative = self._forward_with_cache(negative_ids, cache, "n")

        positive_distance = float(np.sum((question - positive) ** 2))
        negative_distance = float(np.sum((question - negative) ** 2))
        loss = max(0.0, margin + positive_distance - negative_distance)
        if loss <= 0:
            return 0.0

        grad_question = 2.0 * (negative - positive)
        grad_positive = 2.0 * (positive - question)
        grad_negative = 2.0 * (question - negative)

        grad_projection = np.zeros_like(self.projection)
        grad_token_embeddings = np.zeros_like(self.token_embeddings)

        for name, grad_output in [("q", grad_question), ("p", grad_positive), ("n", grad_negative)]:
            self._backward(cache[name], grad_output, grad_projection, grad_token_embeddings)

        self.projection -= learning_rate * grad_projection
        self.token_embeddings -= learning_rate * grad_token_embeddings
        self.token_embeddings[self.pad_id] = 0.0
        return loss

    def _forward_with_cache(self, token_ids, cache, name):
        ids = np.asarray(token_ids, dtype=np.int64)
        mask = ids != self.pad_id
        active_ids = ids[mask]
        if len(active_ids) == 0:
            pooled = np.zeros((self.token_embeddings.shape[1],), dtype=np.float32)
        else:
            pooled = self.token_embeddings[active_ids].mean(axis=0)
        raw = pooled @ self.projection
        output = l2_normalize(raw).astype(np.float32)
        cache[name] = {"active_ids": active_ids, "pooled": pooled, "raw": raw, "output": output}
        return output

    def _backward(self, cache, grad_output, grad_projection, grad_token_embeddings):
        output = cache["output"]
        raw_norm = max(float(np.linalg.norm(cache["raw"])), EPSILON)
        grad_raw = (grad_output - output * float(np.dot(output, grad_output))) / raw_norm
        grad_projection += np.outer(cache["pooled"], grad_raw)
        grad_pooled = self.projection @ grad_raw
        active_ids = cache["active_ids"]
        if len(active_ids) == 0:
            return
        share = grad_pooled / len(active_ids)
        for token_id in active_ids:
            grad_token_embeddings[token_id] += share

