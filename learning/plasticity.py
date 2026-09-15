from dataclasses import dataclass

import numpy as np


@dataclass
class PlasticityUpdate:
    """
    Stores the synaptic updates calculated by a learning rule.

    Attributes
    ----------
    delta_w1 : np.ndarray
        Update to the first weight matrix W1.

    delta_w2 : np.ndarray
        Optional update to W2.

        W2 is currently kept configurable because the final
        experimental architecture has not yet been fixed.
    """

    delta_w1: np.ndarray
    delta_w2: np.ndarray


class ContinuousPlasticityRule:
    """
    Continuous two-parameter learning rule.

    The W1 update follows the project master equation:

        ΔW1 =
            gamma * W2.T @ (Y - Y_hat).T @ X
            + ΔW1_HEBB(eta)

    The Hebbian component uses the Cao et al. formulation:

        eta > 0:
            Oja-style normalized Hebbian learning

        eta < 0:
            normalized anti-Hebbian learning

        eta = 0:
            no Hebbian component

    Parameters
    ----------
    gamma : float
        Strength and polarity of the error-driven component.

    eta : float
        Strength and polarity of the Hebbian component.

    learning_rate : float
        Overall learning rate applied to the calculated update.

    gradient_clip : float or None
        Optional element-wise clipping threshold.
    """

    def __init__(
        self,
        gamma: float,
        eta: float,
        learning_rate: float,
        gradient_clip: float | None = None,
        update_w2: bool = False,
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
        Calculate the complete W1 plasticity update.

        Important interpretation: with ``gamma=1`` and ``eta=0`` in this
        linear feedforward student, the error-feedback term is exactly the
        negative MSE gradient for W1.  It is consequently an
        error-feedback surrogate for Contrastive Hebbian Learning (CHL), not
        a two-phase recurrent/energy-based CHL implementation.  The project
        keeps this condition as the proposal's CHL-labelled comparator, but
        results must not be presented as distinguishing it from GD under the
        current architecture.

        Parameters
        ----------
        x : np.ndarray
            Input batch.

            Shape:
                (batch_size, input_dim)

        y : np.ndarray
            Target outputs.

            Shape:
                (batch_size,)

        h : np.ndarray
            Feedforward cortical activity.

            Shape:
                (batch_size, hidden_dim)

        y_hat : np.ndarray
            Student predictions.

            Shape:
                (batch_size,)

        w1 : np.ndarray
            First weight matrix.

            Shape:
                (hidden_dim, input_dim)

        w2 : np.ndarray
            Second weight matrix.

            Shape:
                (1, hidden_dim)

        Returns
        -------
        PlasticityUpdate
            Calculated updates to W1 and W2.

            W2 is currently zero because its plasticity has
            deliberately not been fixed yet.
        """

        self._validate_inputs(
            x=x,
            y=y,
            h=h,
            y_hat=y_hat,
            w1=w1,
            w2=w2,
        )

        batch_size = x.shape[0]

        # -----------------------------------------------------
        # 1. Prediction error
        #
        # The project architecture defines:
        #
        #     e = y - y_hat
        #
        # -----------------------------------------------------

        error = y - y_hat

        # -----------------------------------------------------
        # 2. Error-driven component
        #
        #     ΔW1_feedback =
        #
        #         gamma * W2.T @ E.T @ X / batch_size
        #
        # Shapes:
        #
        # W2.T  = (hidden_dim, 1)
        # E.T   = (1, batch_size)
        # X     = (batch_size, input_dim)
        #
        # Result:
        #
        # (hidden_dim, input_dim)
        # -----------------------------------------------------

        delta_w1_feedback = (
            self.gamma
            * w2.T
            @ error[np.newaxis, :]
            @ x
        )

        delta_w1_feedback /= batch_size

        # -----------------------------------------------------
        # 3. Hebbian component
        # -----------------------------------------------------

        delta_w1_hebb = self._hebbian_update(
            x=x,
            h=h,
            w1=w1,
        )

        # -----------------------------------------------------
        # 4. Combine both components
        #
        #     ΔW1 =
        #          error-driven component
        #        + Hebbian component
        # -----------------------------------------------------

        delta_w1 = (
            delta_w1_feedback
            + delta_w1_hebb
        )

        # -----------------------------------------------------
        # 5. Optional clipping
        # -----------------------------------------------------

        if self.gradient_clip is not None:
            delta_w1 = np.clip(
                delta_w1,
                -self.gradient_clip,
                self.gradient_clip,
            )

        # -----------------------------------------------------
        # 6. Apply learning rate and output plasticity
        # -----------------------------------------------------

        delta_w1 *= self.learning_rate

        # W2 MUST learn via standard supervised prediction error, 
        # completely independent of the W1 gamma parameter.
        if self.update_w2:
            delta_w2 = self.learning_rate * (
                error[np.newaxis, :] @ h
            ) / batch_size
            
            if self.gradient_clip is not None:
                delta_w2 = np.clip(delta_w2, -self.gradient_clip, self.gradient_clip)
        else:
            delta_w2 = np.zeros_like(w2)

        return PlasticityUpdate(
            delta_w1=delta_w1,
            delta_w2=delta_w2,
        )

    def _hebbian_update(
        self,
        x: np.ndarray,
        h: np.ndarray,
        w1: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the Hebbian/Anti-Hebbian component.

        For eta > 0, use the Oja-style rule:

            Δw_n =
                eta * h_n *
                (x - h_n w_n)

        For eta < 0, use:

            Δw_n =
                eta * h_n x
                / (1 + ||w_n||²)

        For eta = 0:

            ΔW = 0
        """

        if self.eta == 0:
            return np.zeros_like(w1)

        batch_size = x.shape[0]

        if self.eta > 0:
            # -------------------------------------------------
            # Positive eta:
            #
            # Oja-style Hebbian learning.
            #
            # First term:
            #
            #     E[h_n x]
            #
            correlation = (
                h.T @ x
            ) / batch_size

            # Second term:
            #
            #     E[h_n^2] W1
            #
            hidden_squared_mean = (
                np.mean(h ** 2, axis=0)
            )

            normalization = (
                hidden_squared_mean[:, np.newaxis]
                * w1
            )

            return self.eta * (
                correlation - normalization
            )

        # -----------------------------------------------------
        # Negative eta:
        #
        # Anti-Hebbian normalized rule.
        # -----------------------------------------------------

        correlation = (
            h.T @ x
        ) / batch_size

        row_norm_squared = np.sum(
            w1 ** 2,
            axis=1,
        )

        normalization = (
            1.0 + row_norm_squared
        )[:, np.newaxis]

        return (
            self.eta
            * correlation
            / normalization
        )

    @staticmethod
    def _validate_inputs(
        x: np.ndarray,
        y: np.ndarray,
        h: np.ndarray,
        y_hat: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
    ) -> None:

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

        if w1.ndim != 2:
            raise ValueError(
                "w1 must be a 2D array."
            )

        if w2.ndim != 2:
            raise ValueError(
                "w2 must be a 2D array."
            )

        batch_size = x.shape[0]
        input_dim = x.shape[1]
        hidden_dim = h.shape[1]

        if y.shape != (batch_size,):
            raise ValueError(
                "y must have shape (batch_size,)."
            )

        if y_hat.shape != (batch_size,):
            raise ValueError(
                "y_hat must have shape (batch_size,)."
            )

        if w1.shape != (
            hidden_dim,
            input_dim,
        ):
            raise ValueError(
                "w1 has incompatible dimensions."
            )

        if w2.shape != (1, hidden_dim):
            raise ValueError(
                "w2 must have shape "
                "(1, hidden_dim)."
            )
