import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.transformer.dataset import build_next_token_examples
from src.transformer.generate import generate_text
from src.transformer.model import TinyTransformerModel, causal_mask, softmax
from src.transformer.storage import load_transformer_artifacts, save_transformer_artifacts
from src.transformer.train import cross_entropy_loss, train_step
from src.transformer.tokenizer import (
    PAD_TOKEN,
    UNK_TOKEN,
    build_transformer_vocabulary,
    decode_token_ids,
    encode_text,
)
from train_gita_tiny_transformer import select_balanced_texts


class GitaTinyTransformerTests(unittest.TestCase):
    def test_vocabulary_keeps_special_tokens_first_and_limits_size(self):
        vocabulary = build_transformer_vocabulary(
            ["Control anger with practice", "Control mind with wisdom"],
            max_size=6,
        )

        self.assertEqual(vocabulary[PAD_TOKEN], 0)
        self.assertEqual(vocabulary[UNK_TOKEN], 1)
        self.assertEqual(len(vocabulary), 6)
        self.assertIn("control", vocabulary)

    def test_encode_text_pads_and_uses_unknown_token(self):
        vocabulary = build_transformer_vocabulary(["control anger"], max_size=10)

        token_ids = encode_text("control desire", vocabulary, max_length=4)

        self.assertEqual(token_ids[0], vocabulary["control"])
        self.assertEqual(token_ids[1], vocabulary[UNK_TOKEN])
        self.assertEqual(token_ids[2:], [vocabulary[PAD_TOKEN], vocabulary[PAD_TOKEN]])

    def test_decode_token_ids_ignores_padding(self):
        vocabulary = build_transformer_vocabulary(["control anger"], max_size=10)
        token_ids = [vocabulary["control"], vocabulary["anger"], vocabulary[PAD_TOKEN]]

        text = decode_token_ids(token_ids, vocabulary)

        self.assertEqual(text, "control anger")

    def test_next_token_examples_use_fixed_context_windows(self):
        vocabulary = build_transformer_vocabulary(["control anger by steady practice"], max_size=20)

        examples = build_next_token_examples(
            ["control anger by steady practice"],
            vocabulary=vocabulary,
            context_length=3,
        )

        self.assertGreaterEqual(len(examples), 4)
        self.assertEqual(examples[0].input_ids, [0, 0, vocabulary["control"]])
        self.assertEqual(examples[0].target_id, vocabulary["anger"])
        self.assertEqual(examples[2].input_ids, [vocabulary["control"], vocabulary["anger"], vocabulary["by"]])
        self.assertEqual(examples[2].target_id, vocabulary["steady"])

    def test_causal_mask_blocks_future_tokens(self):
        mask = causal_mask(context_length=4)

        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(np.all(mask[np.triu_indices(4, k=1)]))
        self.assertFalse(np.any(mask[np.tril_indices(4)]))

    def test_softmax_returns_probabilities_per_row(self):
        probabilities = softmax(np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]))

        np.testing.assert_allclose(probabilities.sum(axis=-1), np.array([1.0, 1.0]))
        self.assertGreater(probabilities[0, 2], probabilities[0, 0])

    def test_tiny_transformer_forward_returns_logits_and_trace(self):
        model = TinyTransformerModel.create(
            vocab_size=12,
            context_length=4,
            d_model=6,
            hidden_size=10,
            seed=11,
        )

        output = model.forward(np.array([[2, 3, 4, 0]]))

        self.assertEqual(output.logits.shape, (1, 4, 12))
        self.assertEqual(output.token_embeddings.shape, (1, 4, 6))
        self.assertEqual(output.position_embeddings.shape, (1, 4, 6))
        self.assertEqual(output.attention_weights.shape, (1, 4, 4))
        self.assertEqual(output.hidden_states.shape, (1, 4, 10))
        np.testing.assert_allclose(output.probabilities.sum(axis=-1), np.ones((1, 4)), atol=1e-6)

    def test_cross_entropy_uses_last_position_next_token_target(self):
        probabilities = np.array(
            [
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.8, 0.1],
                ]
            ]
        )

        loss = cross_entropy_loss(probabilities, np.array([1]))

        self.assertAlmostEqual(loss, -np.log(0.8), places=6)

    def test_train_step_reduces_loss_for_tiny_batch(self):
        model = TinyTransformerModel.create(
            vocab_size=8,
            context_length=3,
            d_model=6,
            hidden_size=10,
            seed=13,
        )
        input_ids = np.array([[0, 2, 3], [0, 2, 3], [0, 2, 3]])
        target_ids = np.array([4, 4, 4])

        first = train_step(model, input_ids, target_ids, learning_rate=0.8)
        second = train_step(model, input_ids, target_ids, learning_rate=0.8)

        self.assertLess(second.loss, first.loss)
        self.assertEqual(second.gradient_shapes["w_output"], model.w_output.shape)

    def test_transformer_artifacts_round_trip(self):
        model = TinyTransformerModel.create(
            vocab_size=8,
            context_length=3,
            d_model=6,
            hidden_size=10,
            seed=17,
        )
        vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1, "control": 2}

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "dpp-gita-tiny-transformer-v1"
            save_transformer_artifacts(
                model_dir=model_dir,
                model_id="dpp-gita-tiny-transformer-v1",
                model=model,
                vocabulary=vocabulary,
                config={"context_length": 3, "d_model": 6, "hidden_size": 10},
                metrics={"final_loss": 1.2},
            )

            loaded_model, loaded_vocabulary, loaded_config, loaded_card = load_transformer_artifacts(model_dir)

        self.assertEqual(loaded_vocabulary["control"], 2)
        self.assertEqual(loaded_config["context_length"], 3)
        self.assertEqual(loaded_card["pretrained_model_used"], False)
        np.testing.assert_allclose(loaded_model.w_output, model.w_output)

    def test_generate_text_returns_tokens_and_trace(self):
        vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1, "control": 2, "anger": 3, "mind": 4}
        model = TinyTransformerModel.create(
            vocab_size=len(vocabulary),
            context_length=3,
            d_model=6,
            hidden_size=10,
            seed=19,
        )
        model.b_output[:] = 0.0
        model.b_output[vocabulary["mind"]] = 5.0

        result = generate_text(
            model,
            vocabulary,
            prompt="control anger",
            max_new_tokens=2,
            temperature=0.0,
        )

        self.assertEqual(result.generated_tokens, ["mind", "mind"])
        self.assertEqual(result.text, "control anger mind mind")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0]["next_token"], "mind")
        self.assertIn("top_tokens", result.steps[0])

    def test_generate_text_does_not_emit_special_tokens(self):
        vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1, "control": 2, "anger": 3}
        model = TinyTransformerModel.create(
            vocab_size=len(vocabulary),
            context_length=3,
            d_model=6,
            hidden_size=10,
            seed=23,
        )
        model.b_output[:] = 0.0
        model.b_output[vocabulary[UNK_TOKEN]] = 10.0
        model.b_output[vocabulary["anger"]] = 5.0

        result = generate_text(model, vocabulary, prompt="control", max_new_tokens=1)

        self.assertEqual(result.generated_tokens, ["anger"])

    def test_generate_text_samples_only_from_top_k_candidates(self):
        vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1, "control": 2, "anger": 3, "mind": 4, "practice": 5}
        model = TinyTransformerModel.create(
            vocab_size=len(vocabulary),
            context_length=3,
            d_model=6,
            hidden_size=10,
            seed=29,
        )
        model.b_output[:] = 0.0
        model.b_output[vocabulary["anger"]] = 4.0
        model.b_output[vocabulary["mind"]] = 3.0
        model.b_output[vocabulary["practice"]] = 2.0
        model.b_output[vocabulary["control"]] = 1.0

        result = generate_text(
            model,
            vocabulary,
            prompt="control",
            max_new_tokens=12,
            temperature=1.0,
            top_k=2,
            seed=31,
        )

        self.assertLessEqual(set(result.generated_tokens), {"anger", "mind"})
        self.assertEqual([item["token"] for item in result.steps[0]["top_tokens"]], ["anger", "mind"])

    def test_generate_text_can_avoid_common_tokens(self):
        vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1, "control": 2, "the": 3, "mind": 4}
        model = TinyTransformerModel.create(
            vocab_size=len(vocabulary),
            context_length=3,
            d_model=6,
            hidden_size=10,
            seed=37,
        )
        model.b_output[:] = 0.0
        model.b_output[vocabulary["the"]] = 5.0
        model.b_output[vocabulary["mind"]] = 3.0

        result = generate_text(
            model,
            vocabulary,
            prompt="control",
            max_new_tokens=1,
            temperature=0.0,
            top_k=2,
            avoid_tokens={"the"},
        )

        self.assertEqual(result.generated_tokens, ["mind"])

    def test_balanced_training_text_selection_keeps_verse_and_qa_texts(self):
        selected = select_balanced_texts(
            verse_texts=["verse one", "verse two", "verse three"],
            qa_texts=["qa one", "qa two", "qa three"],
            max_texts=4,
        )

        self.assertEqual(len(selected), 4)
        self.assertIn("verse one", selected)
        self.assertIn("qa one", selected)

    def test_balanced_training_text_selection_spreads_across_qa_rows(self):
        selected = select_balanced_texts(
            verse_texts=["verse one", "verse two"],
            qa_texts=[f"qa {index}" for index in range(10)],
            max_texts=6,
        )

        self.assertIn("qa 0", selected)
        self.assertIn("qa 9", selected)


if __name__ == "__main__":
    unittest.main()
