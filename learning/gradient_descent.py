from dataclasses import dataclass

import numpy as np


@dataclass
class GradientUpdate:
    """
    Stores the gradients calculated for the student network.
    """

    grad_w1: np.ndarray
    grad_w2: np.ndarray


def mean_squared_error(
    y: np.ndarray,
    y_hat: np.ndarray,
) -> float:
    """
    Calculate mean squared error with a 1/2 factor.

    L = 1/(2N) * sum((y - y_hat)^2)

    Parameters
    ----------
    y : np.ndarray
        Target values with shape (batch_size,).

    y_hat : np.ndarray
        Predicted values with shape (batch_size,).

    Returns
    -------
    float
        Mean squared error.
    """

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    if y_hat.ndim != 1:
        raise ValueError("y_hat must be a 1D array.")

    if y.shape != y_hat.shape:
        raise ValueError(
            "y and y_hat must have the same shape."
        )

    error = y_hat - y

    return 0.5 * np.mean(error ** 2)


def calculate_gradients(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    y_hat: np.ndarray,
    w2: np.ndarray,
) -> GradientUpdate:
    """
    Calculate the gradients of the student network.

    The student is:

        h = W1 x
        y_hat = W2 h

    For a batch:

        H = X W1.T
        Y_hat = H W2.T

    The gradients are:

        dL/dW2 = (1/N) E.T H

        dL/dW1 = (1/N) W2.T E.T X

    Parameters
    ----------
    x : np.ndarray
        Input batch with shape (batch_size, input_dim).

    y : np.ndarray
        Target values with shape (batch_size,).

    h : np.ndarray
        Hidden states with shape (batch_size, hidden_dim).

    y_hat : np.ndarray
        Predictions with shape (batch_size,).

    w2 : np.ndarray
        Second weight matrix with shape (1, hidden_dim).

    Returns
    -------
    GradientUpdate
        Gradients for W1 and W2.
    """

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    if h.ndim != 2:
        raise ValueError("h must be a 2D array.")

    if y_hat.ndim != 1:
        raise ValueError("y_hat must be a 1D array.")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y must contain the same number of examples."
        )

    if h.shape[0] != x.shape[0]:
        raise ValueError(
            "x and h must contain the same number of examples."
        )

    if y_hat.shape[0] != x.shape[0]:
        raise ValueError(
            "x and y_hat must contain the same number of examples."
        )

    if w2.shape != (1, h.shape[1]):
        raise ValueError(
            "w2 must have shape (1, hidden_dim)."
        )

    batch_size = x.shape[0]

    # Conventional gradient uses:
    #
    # error = prediction - target
    #
    # This is different in sign from the research
    # equation's (y - y_hat), because gradient descent
    # subsequently subtracts the gradient.
    error = y_hat - y

    # ---------------------------------------------------------
    # Gradient with respect to W2
    #
    # E.T @ H
    #
    # (1, batch_size) @ (batch_size, hidden_dim)
    # =
    # (1, hidden_dim)
    # ---------------------------------------------------------

    grad_w2 = (
        error[np.newaxis, :] @ h
    ) / batch_size

    # ---------------------------------------------------------
    # Gradient with respect to W1
    #
    # W2.T @ E.T @ X
    #
    # (hidden_dim, 1)
    #      @
    # (1, batch_size)
    #      @
    # (batch_size, input_dim)
    # =
    # (hidden_dim, input_dim)
    # ---------------------------------------------------------

    grad_w1 = (
        w2.T
        @ error[np.newaxis, :]
        @ x
    ) / batch_size

    return GradientUpdate(
        grad_w1=grad_w1,
        grad_w2=grad_w2,
    )


def apply_gradient_descent(
    w1: np.ndarray,
    w2: np.ndarray,
    gradients: GradientUpdate,
    learning_rate: float,
    update_w2: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply one gradient-descent update.

    Parameters
    ----------
    w1 : np.ndarray
        First weight matrix.

    w2 : np.ndarray
        Second weight matrix.

    gradients : GradientUpdate
        Calculated gradients.

    learning_rate : float
        Learning rate.

    update_w2 : bool
        Whether W2 should be updated.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated W1 and W2.
    """

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than 0."
        )

    new_w1 = w1 - learning_rate * gradients.grad_w1

    if update_w2:
        new_w2 = w2 - learning_rate * gradients.grad_w2
    else:
        new_w2 = w2.copy()

    return new_w1, new_w2