from dataclasses import dataclass

import numpy as np


@dataclass
class WeightUpdate:
    """
    Stores the weight changes calculated by a learning rule.
    """

    delta_w1: np.ndarray
    delta_w2: np.ndarray


class PlasticityRule:
    """
    Configurable learning-rule implementation for the student.

    Parameters
    ----------
    gamma : float
        Strength and polarity of the error-driven feedback component.

    eta : float
        Strength of the Hebbian component.

    learning_rate : float
        Overall size of the weight update.

    update_w2 : bool
        Whether W2 is allowed to change.

    gradient_clip : float or None
        Optional maximum absolute value used to clip updates.
    """

    def __init__(
        self,
        gamma: float,
        eta: float,
        learning_rate: float,
        update_w2: bool = False,
        gradient_clip: float | None = None,
    ):
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0."
            )

        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError(
                "gradient_clip must be greater than 0."
            )

        self.gamma = gamma
        self.eta = eta
        self.learning_rate = learning_rate
        self.update_w2 = update_w2
        self.gradient_clip = gradient_clip

    def calculate_update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        h: np.ndarray,
        y_hat: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
    ) -> WeightUpdate:
        """
        Calculate the weight updates for a batch.

        Parameters
        ----------
        x : np.ndarray
            Input batch with shape (batch_size, input_dim).

        y : np.ndarray
            Target outputs with shape (batch_size,).

        h : np.ndarray
            Hidden/cortical activity with shape
            (batch_size, hidden_dim).

        y_hat : np.ndarray
            Student predictions with shape (batch_size,).

        w1 : np.ndarray
            First weight matrix.

        w2 : np.ndarray
            Second weight matrix.

        Returns
        -------
        WeightUpdate
            Calculated changes to W1 and W2.
        """

        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if h.ndim != 2:
            raise ValueError("h must be a 2D array.")

        if y_hat.ndim != 1:
            raise ValueError(
                "y_hat must be a 1D array."
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "x and y must contain the same number of examples."
            )

        if x.shape[0] != h.shape[0]:
            raise ValueError(
                "x and h must contain the same number of examples."
            )

        if y.shape[0] != y_hat.shape[0]:
            raise ValueError(
                "y and y_hat must contain the same number of examples."
            )

        # -----------------------------------------------------
        # 1. Calculate prediction error
        #
        # e = y - y_hat
        # -----------------------------------------------------

        error = y - y_hat

        # -----------------------------------------------------
        # 2. Error-driven feedback term
        #
        # For a batch:
        #
        # ΔW1_feedback =
        #     gamma * W2.T @ E @ X
        #
        # where E and X are arranged so that the resulting
        # matrix has shape (hidden_dim, input_dim).
        # -----------------------------------------------------

        error_signal = self.gamma * error

        delta_w1_feedback = (
            w2.T @ error_signal[:, np.newaxis].T
        )

        delta_w1_feedback = (
            delta_w1_feedback @ x
        )

        # -----------------------------------------------------
        # 3. Hebbian component
        #
        # This will be replaced with the final Cao/Oja
        # formulation when we implement the full rule.
        #
        # For now the component is explicitly separated so
        # that the architecture of the master equation is
        # visible in the code.
        # -----------------------------------------------------

        delta_w1_hebb = (
            self.eta
            * (h.T @ x)
            / x.shape[0]
        )

        # -----------------------------------------------------
        # 4. Combine learning components
        # -----------------------------------------------------

        delta_w1 = (
            delta_w1_feedback / x.shape[0]
            + delta_w1_hebb
        )

        # -----------------------------------------------------
        # 5. W2 update
        #
        # W2 is optional while we determine the final
        # experimental architecture.
        # -----------------------------------------------------

        if self.update_w2:
            delta_w2 = (
                error[:, np.newaxis].T @ h
            ) / x.shape[0]

        else:
            delta_w2 = np.zeros_like(w2)

        # -----------------------------------------------------
        # 6. Optional clipping
        # -----------------------------------------------------

        if self.gradient_clip is not None:
            delta_w1 = np.clip(
                delta_w1,
                -self.gradient_clip,
                self.gradient_clip,
            )

            delta_w2 = np.clip(
                delta_w2,
                -self.gradient_clip,
                self.gradient_clip,
            )

        # -----------------------------------------------------
        # 7. Apply learning rate
        # -----------------------------------------------------

        delta_w1 *= self.learning_rate
        delta_w2 *= self.learning_rate

        return WeightUpdate(
            delta_w1=delta_w1,
            delta_w2=delta_w2,
        )