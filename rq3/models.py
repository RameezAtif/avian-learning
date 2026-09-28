"""Linear student with vector output for the hierarchical tree."""

import numpy as np


class VectorStudent:
    """
    x -> W1 -> h -> W2 -> y_hat

    x:     (B, input_dim)
    h:     (B, hidden_dim)
    y_hat: (B, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, seed=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(input_dim),
                             (hidden_dim, input_dim))
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim),
                             (output_dim, hidden_dim))

    def forward(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        h = x @ self.W1.T
        y_hat = h @ self.W2.T
        return h, y_hat