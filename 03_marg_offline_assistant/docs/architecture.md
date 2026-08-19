# Marg Offline Architecture

```text
Marg Desktop App
  -> Tkinter desktop window
  -> backend/rag_engine.py
  -> local packaged model files
  -> local CPU answer
```

Marg v1 intentionally serves an already-trained model. Training remains in:

```text
02_gita_ai_models
```

The product app is independent:

```text
03_marg_offline_assistant
```

No runtime dependency on the training folder is required.
