"""
Cao et al. (2020) 8-object binary tree dataset.

Structure:
    Root
    ├── Plant
    │   ├── Tree  → Oak, Pine
    │   └── Flower → Rose, Daisy
    └── Animal
        ├── Bird  → Robin, Canary
        └── Fish  → Sunfish, Salmon

Each object has:
  - Input: one-hot over 8 objects
  - Output: 8-dim semantic feature vector
      dim 0: root (always 1)
      dim 1: Plant(1) / Animal(0)
      dim 2: Tree(1) / Flower(0)  (only meaningful for plants)
      dim 3: Bird(1) / Fish(0)    (only meaningful for animals)
      dim 4: Oak(1) / Pine(0)
      dim 5: Rose(1) / Daisy(0)
      dim 6: Robin(1) / Canary(0)
      dim 7: Sunfish(1) / Salmon(0)

For the avian narrative: the tree structure models hierarchical
song categorization (own-species vs other, syllable-type, individual
exemplar). The specific object names are placeholders.
"""

import numpy as np


OBJECT_NAMES = [
    "Oak", "Pine", "Rose", "Daisy",
    "Robin", "Canary", "Sunfish", "Salmon",
]

CATEGORY_LABELS = {
    "level0": np.zeros(8, dtype=int),                          # all same
    "level1": np.array([0, 0, 0, 0, 1, 1, 1, 1]),               # Plant / Animal
    "level2": np.array([0, 0, 1, 1, 2, 2, 3, 3]),               # Tree/Flower/Bird/Fish
    "level3": np.arange(8),                                     # individual objects
}


def generate_tree_dataset():
    """Returns (X, Y) with shapes (8, 8) each."""
    X = np.eye(8, dtype=float)

    Y = np.zeros((8, 8))
    Y[:, 0] = 1.0
    Y[:4, 1] = 1.0          # Plant
    Y[0:2, 2] = 1.0         # Tree
    Y[2:4, 2] = 0.0         # Flower
    Y[4:6, 3] = 1.0         # Bird
    Y[6:8, 3] = 0.0         # Fish
    Y[0, 4] = 1.0           # Oak
    Y[2, 5] = 1.0           # Rose
    Y[4, 6] = 1.0           # Robin
    Y[6, 7] = 1.0           # Sunfish

    return X, Y


def add_input_noise(X, Y, noise_std, rng):
    """Return a noisy copy of the dataset."""
    X_noisy = X + rng.normal(0.0, noise_std, X.shape)
    return X_noisy, Y.copy()


class SimpleNotebook:
    """Stores (x, y) pairs and samples uniformly on replay."""

    def __init__(self):
        self._x = None
        self._y = None

    def encode(self, x, y):
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)

    def __len__(self):
        return 0 if self._x is None else len(self._x)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self), size=batch_size)
        return self._x[idx], self._y[idx]