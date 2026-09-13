from dataclasses import dataclass

import numpy as np


@dataclass
class TeacherDataset:
    """
    Stores a batch of synthetic teacher-generated experiences.

    Attributes
    ----------
    x : np.ndarray
        Input patterns with shape (num_examples, input_dim).
    y : np.ndarray
        Scalar teacher outputs with shape (num_examples,).
    teacher_weights : np.ndarray
        Fixed teacher weight vector with shape (input_dim,).
    noise : np.ndarray
        Noise added to each teacher output.
    snr : float
        Requested theoretical signal-to-noise ratio.
    signal_variance : float
        Empirical variance of the noiseless teacher signal.
    noise_variance : float
        Empirical variance of the sampled noise.
    """

    x: np.ndarray
    y: np.ndarray
    teacher_weights: np.ndarray
    noise: np.ndarray
    snr: float
    signal_variance: float
    noise_variance: float


def generate_teacher_dataset(
    num_examples: int,
    input_dim: int,
    snr: float,
    seed: int | None = None,
) -> TeacherDataset:
    """
    Generate synthetic teacher data following the setup used by Sun (2023).

    The teacher is:

        y = w^T x + epsilon

    where:

        x_i ~ N(0, 1 / input_dim)
        w_i ~ N(0, sigma_w^2)
        epsilon ~ N(0, sigma_epsilon^2)

    and:

        SNR = sigma_w^2 / sigma_epsilon^2

    with:

        sigma_w^2 + sigma_epsilon^2 = 1
    """

    if num_examples <= 0:
        raise ValueError("num_examples must be greater than 0.")

    if input_dim <= 0:
        raise ValueError("input_dim must be greater than 0.")

    if snr < 0:
        raise ValueError("snr must be non-negative or np.inf.")

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Convert the requested SNR into signal/noise variances
    # ---------------------------------------------------------

    if np.isinf(snr):
        sigma_w_squared = 1.0
        sigma_epsilon_squared = 0.0
    else:
        sigma_w_squared = snr / (snr + 1.0)
        sigma_epsilon_squared = 1.0 / (snr + 1.0)

    sigma_w = np.sqrt(sigma_w_squared)
    sigma_epsilon = np.sqrt(sigma_epsilon_squared)

    # ---------------------------------------------------------
    # 2. Generate the fixed teacher weights
    # ---------------------------------------------------------

    teacher_weights = rng.normal(
        loc=0.0,
        scale=sigma_w,
        size=input_dim,
    )

    # ---------------------------------------------------------
    # 3. Generate input patterns
    #
    #    Sun uses:
    #
    #        x_i ~ N(0, 1/N)
    # ---------------------------------------------------------

    x = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(input_dim),
        size=(num_examples, input_dim),
    )

    # ---------------------------------------------------------
    # 4. Generate the noiseless teacher signal
    # ---------------------------------------------------------

    signal = x @ teacher_weights

    # ---------------------------------------------------------
    # 5. Generate Gaussian output noise
    # ---------------------------------------------------------

    noise = rng.normal(
        loc=0.0,
        scale=sigma_epsilon,
        size=num_examples,
    )

    # ---------------------------------------------------------
    # 6. Produce the teacher output
    # ---------------------------------------------------------

    y = signal + noise

    # ---------------------------------------------------------
    # 7. Measure empirical statistics
    # ---------------------------------------------------------

    signal_variance = np.var(signal)
    noise_variance = np.var(noise)

    return TeacherDataset(
        x=x,
        y=y,
        teacher_weights=teacher_weights,
        noise=noise,
        snr=snr,
        signal_variance=signal_variance,
        noise_variance=noise_variance,
    )