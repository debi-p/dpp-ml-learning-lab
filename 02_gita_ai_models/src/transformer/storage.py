import json
from pathlib import Path

import numpy as np

from src.transformer.model import TinyTransformerModel


def save_transformer_artifacts(model_dir, model_id, model, vocabulary, config, metrics):
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output / "model.npz",
        token_embedding=model.token_embedding,
        position_embedding=model.position_embedding,
        w_q=model.w_q,
        w_k=model.w_k,
        w_v=model.w_v,
        w_attention_out=model.w_attention_out,
        w_hidden=model.w_hidden,
        b_hidden=model.b_hidden,
        w_output=model.w_output,
        b_output=model.b_output,
    )
    (output / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "model_card.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "type": "tiny_transformer_from_scratch",
                "algorithm": "token embeddings + positional embeddings + causal self-attention + feed-forward + next-token cross-entropy",
                "pretrained_model_used": False,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_transformer_artifacts(model_dir):
    root = Path(model_dir)
    weights = np.load(root / "model.npz")
    model = TinyTransformerModel(
        token_embedding=weights["token_embedding"].astype(np.float32),
        position_embedding=weights["position_embedding"].astype(np.float32),
        w_q=weights["w_q"].astype(np.float32),
        w_k=weights["w_k"].astype(np.float32),
        w_v=weights["w_v"].astype(np.float32),
        w_attention_out=weights["w_attention_out"].astype(np.float32),
        w_hidden=weights["w_hidden"].astype(np.float32),
        b_hidden=weights["b_hidden"].astype(np.float32),
        w_output=weights["w_output"].astype(np.float32),
        b_output=weights["b_output"].astype(np.float32),
    )
    vocabulary = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    card = json.loads((root / "model_card.json").read_text(encoding="utf-8"))
    return model, vocabulary, config, card
