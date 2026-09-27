"""
Contrastive Hebbian Learning implemented directly from Cao et al. (2020),
equations (3) and (4).

Cao's two-phase rule uses hidden states:

    h^f = W1 x + γ W2^T ŷ
    h^c = W1 x + γ W2^T y

The updates are:

    ΔW1_CHL = (h^c x^T − h^f x^T) / γ
            = W2^T (y − ŷ) x^T                              [eq 3]

    ΔW2_CHL = y h^{cT} − ŷ h^{fT}
            = (y − ŷ) x^T W1^T + γ (y y^T − ŷ ŷ^T) W2      [eq 4]

Equation (3) shows that the CHL update to W1 is identical to the
gradient-descent update to W1. The rules differ only in:
  - the correlation term of the W2 update (eq 4)
  - the Hebbian term on W1 (eq 5), if eta is non-zero.
"""

from __future__ import annotations

import numpy as np

from learning.plasticity import PlasticityUpdate


class ContrastiveHebbianRule:
    """
    Cao et al. (2020) CHL rule, equations (3) and (4).

    Parameters
    ----------
    learning_rate : float
        Update step size λ.
    gamma : float
        Top-down coupling strength in the hidden-state definition.
        Note that W1's update does not depend on gamma — the gamma
        cancels out in equation (3).
    eta : float
        Hebbian coefficient for the W1 update (equation 5).
        eta = 0 gives pure CHL.
    gradient_clip : float or None
        Optional element-wise clipping threshold.
    update_w2 : bool
        Whether to update W2 with the CHL contrastive rule.
    """

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

        # Populated by the last call to calculate_update.
        # Used by the diagnostic script.
        self.last_terms = None

    def calculate_update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        h: np.ndarray,
        y_hat: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
    ) -> PlasticityUpdate:
        """
        h    : (B, D_h)   bottom-up hidden state = W1 x
        y    : (B,)       targets
        y_hat: (B,)       free-phase output = W2 h
        w1   : (D_h, D_x)
        w2   : (1, D_h)
        """
        B = x.shape[0]
        gamma = self.gamma

        error = y - y_hat

        # ---------------- W1 update (equation 3) ----------------
        # ΔW1 = W2^T (y − ŷ) x^T / B
        delta_w1_chl = (w2.T @ error[None, :] @ x) / B

        # ---------------- W2 update (equation 4) ----------------
        # Term A: (y − ŷ) x^T W1^T, averaged over batch → shape (1, D_h)
        # Equivalent to: error-weighted mean of h. 
        term_A = (error @ h)[None, :] / B

        # Term B: γ (y y^T − ŷ ŷ^T) W2, averaged over batch → shape (1, D_h)
        # With scalar output, y y^T reduces to y^2 per sample.
        corr_coefficient = gamma * (
            np.sum(y ** 2) - np.sum(y_hat ** 2)
        ) / B
        term_B = corr_coefficient * w2

        delta_w2_chl = term_A + term_B

        # ---------------- Hebbian on W1 (equation 5) ------------
        delta_w1_hebb = self._hebbian_update(x, h, w1)

        # ---------------- Combine ------------------------------
        delta_w1 = self.learning_rate * (delta_w1_chl + delta_w1_hebb)
        delta_w2 = self.learning_rate * delta_w2_chl if self.update_w2 \
                   else np.zeros_like(w2)

        if self.gradient_clip is not None:
            delta_w1 = np.clip(delta_w1, -self.gradient_clip, self.gradient_clip)
            if self.update_w2:
                delta_w2 = np.clip(delta_w2, -self.gradient_clip, self.gradient_clip)

        # Store for diagnostics.
        self.last_terms = {
            "delta_w1_chl": delta_w1_chl,
            "delta_w1_hebb": delta_w1_hebb,
            "delta_w2_termA_error": term_A,
            "delta_w2_termB_correlation": term_B,
            "corr_coefficient": corr_coefficient,
        }

        return PlasticityUpdate(delta_w1=delta_w1, delta_w2=delta_w2)

    def _hebbian_update(
        self,
        x: np.ndarray,
        h: np.ndarray,
        w1: np.ndarray,
    ) -> np.ndarray:
        """Equation 5 from Cao et al."""
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