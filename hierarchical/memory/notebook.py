"""
Simple stored-memory notebook for replay.

Stores (x, y) pairs. Replay samples uniformly at random.
"""

import numpy as np


class StoredNotebook:
    def __init__(self):
        self._x = None
        self._y = None

    def encode_batch(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)

    def __len__(self) -> int:
        return 0 if self._x is None else len(self._x)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        if len(self) == 0:
            raise RuntimeError("Notebook is empty.")

        indices = rng.integers(0, len(self), size=batch_size)
        return self._x[indices], self._y[indices]