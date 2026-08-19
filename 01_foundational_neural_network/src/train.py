import numpy as np


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0.0] = 1.0
    total = np.sum(counts)
    return total / (num_classes * counts)


def train_test_split(inputs, labels, test_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(inputs))
    rng.shuffle(indices)

    test_count = max(1, int(len(indices) * test_ratio))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return inputs[train_indices], labels[train_indices], inputs[test_indices], labels[test_indices]


def train_model(model, X_train, y_train, epochs=50, learning_rate=0.1, log_every=10, class_weights=None):
    history = []
    for epoch in range(1, epochs + 1):
        loss = model.train_step(X_train, y_train, learning_rate, class_weights=class_weights)
        history.append(loss)
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"Epoch {epoch:03d} loss: {loss:.4f}")
    return history
