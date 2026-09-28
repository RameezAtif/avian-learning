"""
Linear student with vector output.

x -> W1 -> h -> W2 -> y_hat

x: (B, input_dim)
h: (B, hidden_dim)
y_hat: (B, output_dim)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VectorStudentOutput:
    h: np.ndarray       # (B, hidden_dim)
    y_hat: np.ndarray   # (B, output_dim)


class VectorStudent:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: int | None = None,
    ):
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("Dimensions must be > 0.")

        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(
            0.0, 1.0 / np.sqrt(input_dim),
            (hidden_dim, input_dim),
        )
        self.W2 = rng.normal(
            0.0, 1.0 / np.sqrt(hidden_dim),
            (output_dim, hidden_dim),
        )

    def forward(self, x: np.ndarray) -> VectorStudentOutput:
        """
        x: (B, input_dim)
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]

        h = x @ self.W1.T          # (B, hidden_dim)
        y_hat = h @ self.W2.T      # (B, output_dim)

        return VectorStudentOutput(h=h, y_hat=y_hat)