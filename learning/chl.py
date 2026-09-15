"""
Two-phase Contrastive Hebbian Learning for the linear feedforward student.

Architecture: x -> W1 -> h -> W2 -> y_hat

Key distinction from the error-feedback surrogate:
  - Free phase uses only bottom-up input.
  - Clamped phase drives the hidden layer with the *target* y
    through the top-down projection W2.T.
  - The W1 update is the difference of the two Hebbian outer products,
    which is NOT the same as the gradient of the MSE loss.
"""

from __future__ import annotations

import numpy as np

from learning.plasticity import PlasticityUpdate


class ContrastiveHebbianRule:
    """
    Two-phase Contrastive Hebbian Learning.

    Free phase:
        h_free = W1 @ x
        y_free = W2 @ h_free

    Clamped phase:
        h_clamped = h_free + alpha * W2.T @ y
        (one-step relaxation; alpha is the feedback strength)

    W1 update:
        delta_w1 = lr * (h_clamped - h_free).T @ x / batch_size

    W2 update (optional):
        delta_w2 = lr * (
            y @ h_clamped.T - y_free @ h_free.T
        ) / batch_size
    """

    def __init__(
        self,
        learning_rate: float,
        feedback_strength: float = 1.0,
        gradient_clip: float | None = None,
        update_w2: bool = False,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")

        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip must be > 0.")

        self.learning_rate = learning_rate
        self.alpha = feedback_strength
        self.gradient_clip = gradient_clip
        self.update_w2 = update_w2

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
        Shapes (matching ContinuousPlasticityRule):
            x      : (batch_size, input_dim)
            y      : (batch_size,)
            h      : (batch_size, hidden_dim)   # = h_free
            y_hat  : (batch_size,)              # = y_free
            w1     : (hidden_dim, input_dim)
            w2     : (1, hidden_dim)
        """
        batch_size = x.shape[0]

        # ---- Free phase (already computed by the student) ----
        h_free = h
        y_free = y_hat

        # ---- Clamped phase ----
        # w2.T          : (hidden_dim, 1)
        # y[None, :]    : (1, batch_size)
        # product       : (hidden_dim, batch_size)
        # transpose     : (batch_size, hidden_dim)  -> matches h_free
        h_clamped = h_free + self.alpha * (w2.T @ y[None, :]).T

        # ---- W1 update: difference of Hebbian outer products ----
        # (h_clamped - h_free).T @ x
        #   (hidden_dim, batch) @ (batch, input_dim)
        #   = (hidden_dim, input_dim)  -> matches w1
        delta_w1 = (
            (h_clamped - h_free).T @ x
        ) / batch_size

        # ---- W2 update: contrastive Hebbian on the readout ----
        if self.update_w2:
            delta_w2 = (
                (y[None, :] @ h_clamped)
                - (y_free[None, :] @ h_free)
            ) / batch_size
        else:
            delta_w2 = np.zeros_like(w2)

        # ---- Optional clipping (same convention as the master rule) ----
        if self.gradient_clip is not None:
            delta_w1 = np.clip(
                delta_w1, -self.gradient_clip, self.gradient_clip
            )
            if self.update_w2:
                delta_w2 = np.clip(
                    delta_w2, -self.gradient_clip, self.gradient_clip
                )

        # ---- Apply learning rate LAST (matches ContinuousPlasticityRule) ----
        delta_w1 *= self.learning_rate
        if self.update_w2:
            delta_w2 *= self.learning_rate

        return PlasticityUpdate(
            delta_w1=delta_w1,
            delta_w2=delta_w2,
        )