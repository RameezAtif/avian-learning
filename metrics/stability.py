from dataclasses import dataclass

import numpy as np


@dataclass
class StabilityResult:
    """
    Stores the result of the windowed stability calculation.

    Attributes
    ----------
    stability_variance : float
        Variance across the window means.

    window_means : np.ndarray
        Mean loss for each window.

    final_segment : np.ndarray
        Loss observations used for the calculation.

    start_index : int
        Index where the final evaluation segment begins.

    window_size : int
        Number of observations in each window.

    num_windows : int
        Number of complete windows used.
    """

    stability_variance: float
    window_means: np.ndarray
    final_segment: np.ndarray
    start_index: int
    window_size: int
    num_windows: int


def calculate_windowed_stability(
    loss_history: np.ndarray,
    final_fraction: float = 0.10,
    num_windows: int = 5,
) -> StabilityResult:
    """
    Calculate the Windowed Stability Variance S.

    The procedure is:

        1. Take the final `final_fraction` of the loss history.
        2. Divide it into `num_windows` complete windows.
        3. Calculate the mean loss in each window.
        4. Calculate the variance of those window means.

    Mathematically:

        μ_k = 1/N sum_i L_{k,i}

        S = 1/K sum_k (μ_k - μ_bar)^2

    Parameters
    ----------
    loss_history : np.ndarray
        One-dimensional loss history.

    final_fraction : float
        Fraction of the loss history to use.

        Default:
            0.10

    num_windows : int
        Number of windows.

    Returns
    -------
    StabilityResult
        Windowed stability statistics.
    """

    loss_history = np.asarray(
        loss_history,
        dtype=float,
    )

    if loss_history.ndim != 1:
        raise ValueError(
            "loss_history must be a 1D array."
        )

    if loss_history.size == 0:
        raise ValueError(
            "loss_history must not be empty."
        )

    if not 0 < final_fraction <= 1:
        raise ValueError(
            "final_fraction must be in the interval (0, 1]."
        )

    if num_windows <= 0:
        raise ValueError(
            "num_windows must be greater than 0."
        )

    if not np.all(np.isfinite(loss_history)):
        raise ValueError(
            "loss_history must contain only finite values."
        )

    # ---------------------------------------------------------
    # 1. Determine the final evaluation segment.
    # ---------------------------------------------------------

    requested_segment_size = max(
        1,
        int(
            np.ceil(
                loss_history.size
                * final_fraction
            )
        ),
    )

    # We need enough observations to make the requested
    # number of windows possible.
    if requested_segment_size < num_windows:
        raise ValueError(
            "The final segment does not contain enough "
            "observations for the requested number of windows."
        )

    start_index = (
        loss_history.size
        - requested_segment_size
    )

    final_segment = loss_history[
        start_index:
    ]

    # ---------------------------------------------------------
    # 2. Determine the number of observations per window.
    # ---------------------------------------------------------

    window_size = (
        final_segment.size // num_windows
    )

    if window_size <= 0:
        raise ValueError(
            "Window size must be greater than 0."
        )

    usable_size = (
        window_size * num_windows
    )

    # Discard any remainder so that every window has exactly
    # the same number of observations.
    final_segment = final_segment[
        :usable_size
    ]

    # ---------------------------------------------------------
    # 3. Reshape into windows.
    #
    # Shape:
    #
    #     (num_windows, window_size)
    # ---------------------------------------------------------

    windows = final_segment.reshape(
        num_windows,
        window_size,
    )

    # ---------------------------------------------------------
    # 4. Calculate each window's mean.
    # ---------------------------------------------------------

    window_means = np.mean(
        windows,
        axis=1,
    )

    # ---------------------------------------------------------
    # 5. Calculate the grand mean of the window means.
    # ---------------------------------------------------------

    grand_mean = np.mean(
        window_means
    )

    # ---------------------------------------------------------
    # 6. Calculate stability variance.
    #
    #     S = 1/K sum(μ_k - μ_bar)^2
    #
    # Use population variance because the equation in the
    # architecture document divides by K.
    # ---------------------------------------------------------

    stability_variance = float(
        np.mean(
            (window_means - grand_mean) ** 2
        )
    )

    return StabilityResult(
        stability_variance=stability_variance,
        window_means=window_means,
        final_segment=final_segment,
        start_index=start_index,
        window_size=window_size,
        num_windows=num_windows,
    )