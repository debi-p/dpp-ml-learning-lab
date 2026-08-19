from dataclasses import dataclass

import numpy as np


@dataclass
class TrainStepResult:
    loss: float
    gradient_shapes: dict


def cross_entropy_loss(probabilities, target_ids):
    probabilities = np.asarray(probabilities)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    batch_size = probabilities.shape[0]
    last_position_probabilities = probabilities[:, -1, :]
    picked = last_position_probabilities[np.arange(batch_size), target_ids]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


def train_step(model, input_ids, target_ids, learning_rate=0.05):
    input_ids = np.asarray(input_ids, dtype=np.int64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    output = model.forward(input_ids)
    batch_size = input_ids.shape[0]
    context_length = model.context_length

    loss = cross_entropy_loss(output.probabilities, target_ids)

    d_logits = np.zeros_like(output.logits)
    d_logits[:, -1, :] = output.probabilities[:, -1, :]
    d_logits[np.arange(batch_size), -1, target_ids] -= 1.0
    d_logits /= batch_size

    hidden_flat = output.hidden_states.reshape(-1, output.hidden_states.shape[-1])
    d_logits_flat = d_logits.reshape(-1, d_logits.shape[-1])
    grad_w_output = hidden_flat.T @ d_logits_flat
    grad_b_output = d_logits_flat.sum(axis=0)

    d_hidden = d_logits @ model.w_output.T
    d_hidden_linear = d_hidden * (output.hidden_linear > 0.0)

    attention_projected_flat = output.attention_projected.reshape(-1, output.attention_projected.shape[-1])
    d_hidden_linear_flat = d_hidden_linear.reshape(-1, d_hidden_linear.shape[-1])
    grad_w_hidden = attention_projected_flat.T @ d_hidden_linear_flat
    grad_b_hidden = d_hidden_linear_flat.sum(axis=0)

    d_attention_projected = d_hidden_linear @ model.w_hidden.T
    attention_output_flat = output.attention_output.reshape(-1, output.attention_output.shape[-1])
    d_attention_projected_flat = d_attention_projected.reshape(-1, d_attention_projected.shape[-1])
    grad_w_attention_out = attention_output_flat.T @ d_attention_projected_flat

    d_attention_output = d_attention_projected @ model.w_attention_out.T
    grad_attention_weights = d_attention_output @ np.swapaxes(output.v, -1, -2)
    d_v = np.swapaxes(output.attention_weights, -1, -2) @ d_attention_output

    d_attention_scores = _softmax_backward(output.attention_weights, grad_attention_weights)
    d_attention_scores[:, np.triu_indices(context_length, k=1)[0], np.triu_indices(context_length, k=1)[1]] = 0.0

    scale = np.sqrt(model.d_model)
    d_q = (d_attention_scores @ output.k) / scale
    d_k = (np.swapaxes(d_attention_scores, -1, -2) @ output.q) / scale

    combined_flat = output.combined_embeddings.reshape(-1, output.combined_embeddings.shape[-1])
    grad_w_q = combined_flat.T @ d_q.reshape(-1, d_q.shape[-1])
    grad_w_k = combined_flat.T @ d_k.reshape(-1, d_k.shape[-1])
    grad_w_v = combined_flat.T @ d_v.reshape(-1, d_v.shape[-1])

    d_combined = d_q @ model.w_q.T
    d_combined += d_k @ model.w_k.T
    d_combined += d_v @ model.w_v.T

    grad_token_embedding = np.zeros_like(model.token_embedding)
    np.add.at(grad_token_embedding, input_ids, d_combined)
    grad_position_embedding = d_combined.sum(axis=0)

    gradients = {
        "token_embedding": grad_token_embedding,
        "position_embedding": grad_position_embedding,
        "w_q": grad_w_q,
        "w_k": grad_w_k,
        "w_v": grad_w_v,
        "w_attention_out": grad_w_attention_out,
        "w_hidden": grad_w_hidden,
        "b_hidden": grad_b_hidden,
        "w_output": grad_w_output,
        "b_output": grad_b_output,
    }

    model.token_embedding -= learning_rate * grad_token_embedding
    model.position_embedding -= learning_rate * grad_position_embedding
    model.w_q -= learning_rate * grad_w_q
    model.w_k -= learning_rate * grad_w_k
    model.w_v -= learning_rate * grad_w_v
    model.w_attention_out -= learning_rate * grad_w_attention_out
    model.w_hidden -= learning_rate * grad_w_hidden
    model.b_hidden -= learning_rate * grad_b_hidden
    model.w_output -= learning_rate * grad_w_output
    model.b_output -= learning_rate * grad_b_output

    return TrainStepResult(
        loss=loss,
        gradient_shapes={name: value.shape for name, value in gradients.items()},
    )


def _softmax_backward(probabilities, upstream):
    dot = np.sum(upstream * probabilities, axis=-1, keepdims=True)
    return probabilities * (upstream - dot)
