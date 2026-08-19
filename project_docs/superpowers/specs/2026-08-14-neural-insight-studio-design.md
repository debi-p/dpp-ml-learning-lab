# Neural Insight Studio Design

## Goal

Build a common visualizer UI that teaches how a selected model processes data internally. For Phase 1, it explains `dpp-email-classifier-small-v1` using the existing from-scratch email classifier.

## Baseline Direction

Use the first deployed mockup style as the baseline:

- Dark professional dashboard.
- Logo and title: `Neural Insight Studio`.
- Model dropdown first.
- Mode buttons: `Training Mode` and `Ask Mode`.
- Cinematic canvas with boxes and animated connection lines.
- No flowing text between boxes.
- Actual values appear in the paused inspection panel.

## Top-Level Layout

Header:

- Product logo.
- Title: `Neural Insight Studio`.
- Subtitle: `Transparent model training and inference flow for learning how neural networks work internally.`
- Model dropdown with registered models.
- Mode buttons:
  - `Training Mode`
  - `Ask Mode`

Control bar:

- Dataset badge for Training Mode.
- Previous icon button.
- Play icon button.
- Pause icon button.
- Next icon button.
- Replay icon button.
- Speed control with slow-to-fast slider.
- Speed label such as `Slow`, `Normal`, or `Fast`.

Main area:

- Left/center cinematic flow canvas.
- Right inspection panel.
- Bottom teaching cards.

## Training Mode

Training Mode uses the built-in email dataset for Phase 1.

Dataset wording:

- Input: built-in email dataset.
- Source: spam and ham emails.
- Output: expected category.
- Categories:
  - Spam Email
  - Personal Email
  - Work Email
  - Promotional Email

The model still predicts for training rows. Prediction is required because loss is calculated by comparing prediction with expected category.

Training row behavior:

```text
dataset row
-> forward pass
-> prediction
-> compare with expected category
-> loss
-> backpropagation
-> weight and bias update
```

Testing row behavior:

```text
dataset row
-> forward pass
-> prediction
-> compare with expected category
-> accuracy only
```

Testing rows must not update weights.

## Ask Mode

Ask Mode uses a message entered by the user.

Ask Mode behavior:

```text
input message
-> forward pass
-> predicted email category
```

Ask Mode does not ask for expected category. It does not show loss, backpropagation, or weight update because the model is only answering.

## Phase 1 Flow Boxes

The Phase 1 model flow uses these boxes:

1. `Input Dataset`
   - Built-in spam and ham email dataset.
2. `Word Tokenization`
   - Clean text and split into word tokens.
3. `Vocabulary Lookup`
   - Match tokens against known model words.
4. `Text Vectorization`
   - Create fixed-size numeric input vector.
5. `Layer 1: Hidden Layer (32 Neurons)`
   - Weighted sum plus bias for each hidden neuron.
6. `ReLU Activation`
   - Negative values become zero; positive values pass through.
7. `Layer 2: Output Layer (4 Neurons)`
   - Scores for Work, Personal, Promotional, and Spam.
8. `Softmax + Prediction`
   - Convert scores into probabilities and pick the highest.
9. `Loss + Backpropagation`
   - Training rows update weights; testing rows measure only.

We may split `Softmax + Prediction` and `Loss + Backpropagation` into separate boxes later if the user wants a more detailed view.

## Inspection Panel

When playback is paused on a box, the side panel shows:

- Current box name.
- Input value entering the box.
- Algorithm used inside the box.
- Output value leaving the box.

Examples:

Tokenization:

```text
Input:
"limited offer claim your reward today"

Algorithm:
lowercase + regex cleanup + whitespace split

Output:
["limited", "offer", "claim", "your", "reward", "today"]
```

Output layer:

```text
Input:
hidden activations

Algorithm:
z = hidden @ W2 + b2

Output:
work: 0.11
personal: -0.04
promotion: 1.28
spam: 0.44
```

## Playback Behavior

Playback controls advance through the trace timeline.

- Previous: move to previous step.
- Play: run steps automatically.
- Pause: stop on current step.
- Next: move to next step.
- Replay: restart current flow.
- Speed: controls delay between steps.

The active box should glow. Connection lines should animate softly. The canvas should not show moving text labels.

## Generic Model Requirement

The visualizer must not hard-code Phase 1 forever. The UI should render from model trace metadata:

- model id
- mode
- nodes
- edges
- timeline steps
- current step details
- metrics

For Phase 1, the API can return a Phase 1-shaped trace. Later, Phase 2 can return a transformer-shaped trace with tokens, embeddings, attention, logits, softmax, and next-token prediction.

## Acceptance Criteria

- User can select a model from dropdown.
- User can switch between Training Mode and Ask Mode.
- Training Mode shows the Phase 1 boxes in the first-deployed visual style.
- Ask Mode uses the same canvas style but stops at prediction.
- The right panel changes when stepping through boxes.
- Playback controls work.
- Speed slider changes playback speed.
- Testing rows clearly show no weight update.
- The canvas does not display flowing text between boxes.
