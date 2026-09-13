from dataclasses import dataclass

import numpy as np


@dataclass
class StudentOutput:
    """
    Stores the outputs produced by the student network.

    Attributes
    ----------
    h_ff : np.ndarray
        Feedforward hidden state with shape
        (num_examples, hidden_dim).

    y_hat : np.ndarray
        Student predictions with shape
        (num_examples,).
    """

    h_ff: np.ndarray
    y_hat: np.ndarray


class Student:
    """
    Linear two-weight-matrix student network.

    The forward pass is:

        h_ff = W1 x

        y_hat = W2 h_ff

    For a batch of inputs:

        h_ff = X @ W1.T

        y_hat = h_ff @ W2.T

    Parameters
    ----------
    input_dim : int
        Number of input features.

    hidden_dim : int
        Number of hidden/cortical units.

    seed : int or None
        Random seed used to initialise the weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seed: int | None = None,
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be greater than 0.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0.")

        rng = np.random.default_rng(seed)

        # W1 maps:
        #
        # input_dim -> hidden_dim
        #
        # Shape:
        # (hidden_dim, input_dim)
        self.W1 = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(input_dim),
            size=(hidden_dim, input_dim),
        )

        # W2 maps:
        #
        # hidden_dim -> scalar output
        #
        # Shape:
        # (1, hidden_dim)
        self.W2 = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(hidden_dim),
            size=(1, hidden_dim),
        )

    def forward(self, x: np.ndarray) -> StudentOutput:
        """
        Perform a forward pass through the student.

        Parameters
        ----------
        x : np.ndarray
            Input data.

            Accepted shapes:

                (input_dim,)
                (num_examples, input_dim)

        Returns
        -------
        StudentOutput
            Feedforward hidden states and predictions.
        """

        if x.ndim == 1:
            if x.shape[0] != self.W1.shape[1]:
                raise ValueError(
                    "Input dimension does not match student input_dim."
                )

            # Single example:
            #
            # W1 @ x
            #
            # gives shape (hidden_dim,)
            h_ff = self.W1 @ x

            # W2 @ h_ff gives shape (1,)
            y_hat = self.W2 @ h_ff

            # Convert scalar output from shape (1,) to scalar-like
            # NumPy array shape ()
            y_hat = y_hat.squeeze()

            return StudentOutput(
                h_ff=h_ff,
                y_hat=y_hat,
            )

        if x.ndim == 2:
            if x.shape[1] != self.W1.shape[1]:
                raise ValueError(
                    "Input dimension does not match student input_dim."
                )

            # Batch:
            #
            # X @ W1.T
            #
            # Shape:
            # (num_examples, input_dim)
            #       @
            # (input_dim, hidden_dim)
            #       =
            # (num_examples, hidden_dim)
            h_ff = x @ self.W1.T

            # h_ff @ W2.T
            #
            # Shape:
            # (num_examples, hidden_dim)
            #       @
            # (hidden_dim, 1)
            #       =
            # (num_examples, 1)
            y_hat = h_ff @ self.W2.T

            # Our teacher outputs are stored as:
            #
            # (num_examples,)
            #
            # so remove the final singleton dimension.
            y_hat = y_hat.squeeze(axis=1)

            return StudentOutput(
                h_ff=h_ff,
                y_hat=y_hat,
            )

        raise ValueError(
            "x must have shape (input_dim,) or "
            "(num_examples, input_dim)."
        )