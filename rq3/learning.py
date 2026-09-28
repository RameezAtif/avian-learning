"""Vector-output learning rules for RQ3."""

import numpy as np


class PlasticityUpdate:
    def __init__(self, delta_w1, delta_w2):
        self.delta_w1 = delta_w1
        self.delta_w2 = delta_w2


class ContinuousPlasticityVector:
    """GD, QPC, Hebbian, Anti-Hebbian with vector output."""

    def __init__(self, gamma, eta, learning_rate,
                 gradient_clip=None, update_w2=True):
        self.gamma = gamma
        self.eta = eta
        self.lr = learning_rate
        self.clip = gradient_clip
        self.update_w2 = update_w2

    def calculate_update(self, x, y, h, y_hat, w1, w2):
        B = x.shape[0]
        error = y - y_hat                                   # (B, K)

        # Error-driven W1 update
        dW1_err = (w2.T @ error.T @ x) / B                  # (D_h, D_in)
        dW1_hebb = self._hebbian(x, h, w1)
        dW1 = self.gamma * dW1_err + dW1_hebb

        if self.clip is not None:
            dW1 = np.clip(dW1, -self.clip, self.clip)
        dW1 *= self.lr

        if self.update_w2:
            dW2 = (error.T @ h) / B                         # (K, D_h)
            if self.clip is not None:
                dW2 = np.clip(dW2, -self.clip, self.clip)
            dW2 *= self.lr
        else:
            dW2 = np.zeros_like(w2)

        return PlasticityUpdate(dW1, dW2)

    def _hebbian(self, x, h, w1):
        if self.eta == 0:
            return np.zeros_like(w1)
        B = x.shape[0]

        if self.eta > 0:
            corr = (h.T @ x) / B
            h_sq = np.mean(h ** 2, axis=0)
            norm = h_sq[:, None] * w1
            return self.eta * (corr - norm)

        corr = (h.T @ x) / B
        row_norm = np.sum(w1 ** 2, axis=1)
        return self.eta * corr / (1.0 + row_norm)[:, None]


class ContrastiveHebbianVector:
    """
    Cao-literal CHL with vector output.

    ΔW1 = W2ᵀ (y − ŷ)ᵀ x / B
    ΔW2 = (y − ŷ)ᵀ h / B + γ (yᵀy − ŷᵀŷ) W2 / B
    """

    def __init__(self, learning_rate, gamma=1.0, eta=0.0,
                 gradient_clip=None, update_w2=True):
        self.lr = learning_rate
        self.gamma = gamma
        self.eta = eta
        self.clip = gradient_clip
        self.update_w2 = update_w2
        self.last_terms = None

    def calculate_update(self, x, y, h, y_hat, w1, w2):
        B = x.shape[0]
        error = y - y_hat

        dW1 = (w2.T @ error.T @ x) / B
        if self.eta != 0:
            dW1 = dW1 + self._hebbian(x, h, w1)

        termA = (error.T @ h) / B
        corr = (y.T @ y - y_hat.T @ y_hat) / B
        # Spectral clipping to keep the matrix bounded
        spec = np.linalg.norm(corr, ord=2)
        if spec > 1.0:
            corr = corr / spec
        termB = self.gamma * (corr @ w2)

        dW2 = termA + termB

        if self.clip is not None:
            dW1 = np.clip(dW1, -self.clip, self.clip)
            dW2 = np.clip(dW2, -self.clip, self.clip)

        dW1 = self.lr * dW1
        if self.update_w2:
            dW2 = self.lr * dW2
        else:
            dW2 = np.zeros_like(w2)

        self.last_terms = {
            "termA": float(np.linalg.norm(termA)),
            "termB": float(np.linalg.norm(termB)),
            "corr_spec": float(spec),
        }

        return PlasticityUpdate(dW1, dW2)

    def _hebbian(self, x, h, w1):
        B = x.shape[0]
        if self.eta > 0:
            corr = (h.T @ x) / B
            h_sq = np.mean(h ** 2, axis=0)
            return self.eta * (corr - h_sq[:, None] * w1)
        corr = (h.T @ x) / B
        row_norm = np.sum(w1 ** 2, axis=1)
        return self.eta * corr / (1.0 + row_norm)[:, None]