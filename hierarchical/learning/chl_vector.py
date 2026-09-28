"""
Cao-literal CHL with vector output.

Equations:
    ΔW1 = W2^T (y − ŷ) x^T / B
    ΔW2 = (y − ŷ)^T h / B + γ (y^T y − ŷ^T ŷ) W2 / B
    ΔW1_HEBB from equation 5.
"""

import numpy as np

from hierarchical.learning.plasticity_vector import VectorPlasticityUpdate


class ContrastiveHebbianVector:
    def __init__(
        self,
        learning_rate: float,
        gamma: float = 1.0,
        eta: float = 0.0,
        gradient_clip: float | None = None,
        update_w2: bool = True,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip must be > 0.")

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eta = eta
        self.gradient_clip = gradient_clip
        self.update_w2 = update_w2
        self.last_terms = None

    def calculate_update(self, x, y, h, y_hat, w1, w2) -> VectorPlasticityUpdate:
        B = x.shape[0]
        error = y - y_hat  # (B, K)

        # ---- W1 (equation 3) ----
        delta_w1_chl = (w2.T @ error.T @ x) / B  # (D_h, D_in)

        # ---- W2 (equation 4) ----
        # Term A: error-driven
        term_A = (error.T @ h) / B  # (K, D_h)

        # Term B: variance-matching
        # corr_matrix = (y^T y − ŷ^T ŷ)/B    (K, K)
        corr_matrix = (y.T @ y - y_hat.T @ y_hat) / B

        # --- Normalization to prevent runaway ---
        # Bound the spectral norm of corr_matrix at 1 so it acts like
        # a bounded gain rather than an amplifier.
        spectral_norm = np.linalg.norm(corr_matrix, ord=2)
        if spectral_norm > 1.0:
            corr_matrix = corr_matrix / spectral_norm

        term_B = self.gamma * (corr_matrix @ w2)  # (K, D_h)

        delta_w2_chl = term_A + term_B

        # ---- Hebbian on W1 ----
        delta_w1_hebb = self._hebbian_update(x, h, w1)
        delta_w1 = self.learning_rate * (delta_w1_chl + delta_w1_hebb)

        if self.update_w2:
            delta_w2 = self.learning_rate * delta_w2_chl
        else:
            delta_w2 = np.zeros_like(w2)

        if self.gradient_clip is not None:
            delta_w1 = np.clip(delta_w1, -self.gradient_clip, self.gradient_clip)
            if self.update_w2:
                delta_w2 = np.clip(delta_w2, -self.gradient_clip, self.gradient_clip)

        self.last_terms = {
            "termA": term_A,
            "termB": term_B,
            "corr_matrix_trace": float(np.trace(corr_matrix)),
            "corr_matrix_frobenius": float(np.linalg.norm(corr_matrix, "fro")),
            "corr_matrix_spectral": float(spectral_norm),
        }

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

        correlation = (h.T @ x) / B
        row_norm_squared = np.sum(w1 ** 2, axis=1)
        normalization = (1.0 + row_norm_squared)[:, np.newaxis]
        return self.eta * correlation / normalization