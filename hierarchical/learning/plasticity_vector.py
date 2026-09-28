"""
Vector-output versions of the continuous plasticity rules.

Same master equation as before:
    ΔW1 = γ W2^T (y − ŷ) x^T + ΔW1_HEBB(η)

but with y and ŷ now being (B, K) vectors rather than (B,) scalars.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VectorPlasticityUpdate:
    delta_w1: np.ndarray
    delta_w2: np.ndarray


class ContinuousPlasticityVector:
    def __init__(
        self,
        gamma: float,
        eta: float,
        learning_rate: float,
        gradient_clip: float | None = None,
        update_w2: bool = True,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip must be > 0.")

        self.gamma = gamma
        self.eta = eta
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.update_w2 = update_w2

    def calculate_update(self, x, y, h, y_hat, w1, w2) -> VectorPlasticityUpdate:
        """
        x      : (B, input_dim)
        y      : (B, output_dim)
        h      : (B, hidden_dim)
        y_hat  : (B, output_dim)
        w1     : (hidden_dim, input_dim)
        w2     : (output_dim, hidden_dim)
        """
        B = x.shape[0]
        error = y - y_hat  # (B, K)

        # Error-driven component
        # W2^T @ error^T @ x
        # (D_h, K) @ (K, B) @ (B, D_in) = (D_h, D_in)
        delta_w1_feedback = (w2.T @ error.T @ x) / B

        # Hebbian component
        delta_w1_hebb = self._hebbian_update(x, h, w1)

        delta_w1 = self.gamma * delta_w1_feedback + delta_w1_hebb

        if self.gradient_clip is not None:
            delta_w1 = np.clip(delta_w1, -self.gradient_clip, self.gradient_clip)

        delta_w1 *= self.learning_rate

        # W2 update (supervised)
        if self.update_w2:
            delta_w2 = (error.T @ h) / B
            if self.gradient_clip is not None:
                delta_w2 = np.clip(delta_w2, -self.gradient_clip, self.gradient_clip)
            delta_w2 *= self.learning_rate
        else:
            delta_w2 = np.zeros_like(w2)

        return VectorPlasticityUpdate(delta_w1=delta_w1, delta_w2=delta_w2)

    def _hebbian_update(self, x, h, w1):
        if self.eta == 0:
            return np.zeros_like(w1)

        B = x.shape[0]

        if self.eta > 0:
            correlation = (h.T @ x) / B
            hidden_squared_mean = np.mean(h ** 2, axis=0)
            normalization = hidden_squared_mean[:, np.newaxis] * w1
            return self.eta * (correlation - normalization)

        # Anti-Hebbian
        correlation = (h.T @ x) / B
        row_norm_squared = np.sum(w1 ** 2, axis=1)
        normalization = (1.0 + row_norm_squared)[:, np.newaxis]
        return self.eta * correlation / normalization