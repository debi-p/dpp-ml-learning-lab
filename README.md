# DPP ML Learning Lab

This repository is a hands-on learning lab for building machine learning systems from scratch and turning them into usable local applications.

The learning path is:

```text
common neural network architecture
-> foundational neural network from scratch
-> Bhagavad Gita AI models from scratch
-> offline Marg desktop assistant
-> common API, registry, testing UI, and visualizer
```

## Common Neural Network Architecture

This is the main mental model for the whole repository: data enters, becomes tokens and vectors, passes through layers, produces output probabilities, calculates loss, then updates weights through backpropagation.

![End-to-End Neural Network Transformer Flow](./00_neural_network_common_architecture/hld.png)

Additional architecture reference:

![Neural Network Architecture Reference](./00_neural_network_common_architecture/hld1.png)

## Main Folders

- `00_neural_network_common_architecture` - Common neural network and transformer architecture diagrams.
- `01_foundational_neural_network` - Email classifier from scratch with tokenization, vectorization, layers, softmax, loss, backpropagation, SDK, and tests.
- `02_gita_ai_models` - Bhagavad Gita retrieval, embedding, RAG, and tiny transformer models built without pretrained models.
- `03_marg_offline_assistant` - Standalone offline macOS desktop app using the packaged local Gita assistant.
- `common_model_api` - Shared FastAPI service for registered local models.
- `model_registry` - JSON catalog of available models and their artifact paths.
- `model_flow_visualizer` - Browser UI for cinematic step-by-step model data flow inspection.
- `model_testing_ui` - Placeholder for a simple common prediction-testing UI.
- `project_docs` - Shared project planning and architecture notes.
- `archive` - Old notebooks, raw data, metadata, and previous working material.

## Repository Goal

The goal is not only to produce models.

The goal is to understand and document:

- how raw data becomes tokens
- how tokens become vectors
- how vectors pass through layers
- how predictions are produced
- how loss is calculated
- how backpropagation updates weights and biases
- how embeddings support semantic retrieval
- how RAG combines retrieval, context, and answer generation
- how a model becomes an SDK, API, UI, and desktop app

## Recommended Reading Order

1. `00_neural_network_common_architecture/README.md`
2. `01_foundational_neural_network/README.md`
3. `model_flow_visualizer/README.md`
4. `common_model_api/README.md`
5. `02_gita_ai_models/README.md`
6. `03_marg_offline_assistant/README.md`
7. `model_registry/README.md`
8. `model_testing_ui/README.md`

## Quick Commands

Phase 1 tests:

```bash
cd /Users/debi.pradhan/Documents/ML/01_foundational_neural_network
python3 run_all_tests.py
```

Gita model tests:

```bash
cd /Users/debi.pradhan/Documents/ML/02_gita_ai_models
python3 run_all_tests.py
```

Marg tests:

```bash
cd /Users/debi.pradhan/Documents/ML/03_marg_offline_assistant
python3 -m unittest discover -s tests
```

Common API:

```bash
cd /Users/debi.pradhan/Documents/ML/common_model_api
uvicorn app:app --reload --port 8010
```

Visualizer:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
python3 serve_ui.py
```

Marg desktop app:

```bash
cd /Users/debi.pradhan/Documents/ML/03_marg_offline_assistant
python3 run_marg_desktop.py
```

## Naming

Model IDs use a stable style:

```text
dpp-email-classifier-small-v1
dpp-gita-embedding-small-v1
dpp-gita-rag-assistant-v2
dpp-gita-tiny-transformer-v1
dpp-marg-intent-small-v1
```

`dpp` means Debi Prasad Pradhan.

## Development Principle

Every phase should end with something runnable and testable.

```text
learn concept
-> build small version
-> test it
-> export model
-> expose through SDK/API/UI/app
```
