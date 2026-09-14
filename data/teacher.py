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
    condition: str = "unspecified"


@dataclass(frozen=True)
class Teacher:
    """A fixed linear mapping shared by all experience conditions."""
    weights: np.ndarray
    signal_variance: float


def create_teacher(input_dim: int, signal_variance: float = 1.0, seed: int | None = None) -> Teacher:
    """Create one teacher; experience noise is deliberately separate."""
    if input_dim <= 0:
        raise ValueError("input_dim must be greater than 0.")
    if signal_variance <= 0:
        raise ValueError("signal_variance must be greater than 0.")
    rng = np.random.default_rng(seed)
    return Teacher(rng.normal(0.0, np.sqrt(signal_variance), input_dim), signal_variance)


def generate_teacher_experiences(teacher: Teacher, num_examples: int, noise_variance: float,
                                 seed: int | None = None, condition: str = "unspecified") -> TeacherDataset:
    """Sample Gaussian experiences from an existing teacher.

    Tutor and practice conditions share teacher weights and differ only in
    output noise, as required by the proposal.
    """
    if num_examples <= 0:
        raise ValueError("num_examples must be greater than 0.")
    if noise_variance < 0:
        raise ValueError("noise_variance must be non-negative.")
    rng = np.random.default_rng(seed)
    input_dim = teacher.weights.size
    x = rng.normal(0.0, 1.0 / np.sqrt(input_dim), (num_examples, input_dim))
    signal = x @ teacher.weights
    noise = rng.normal(0.0, np.sqrt(noise_variance), num_examples)
    snr = np.inf if noise_variance == 0 else teacher.signal_variance / noise_variance
    return TeacherDataset(x, signal + noise, teacher.weights.copy(), noise, float(snr),
                          float(np.var(signal)), float(np.var(noise)), condition)


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

    # ---------------------------------------------------------
    # 1. Convert the requested SNR into signal/noise variances
    # ---------------------------------------------------------

    if np.isinf(snr):
        sigma_w_squared = 1.0
        sigma_epsilon_squared = 0.0
    else:
        sigma_w_squared = snr / (snr + 1.0)
        sigma_epsilon_squared = 1.0 / (snr + 1.0)

    # Backwards-compatible one-condition generator. New experiments should
    # create one Teacher and sample multiple experience conditions from it.
    teacher = create_teacher(input_dim, sigma_w_squared, seed)
    return generate_teacher_experiences(teacher, num_examples, sigma_epsilon_squared,
                                        None if seed is None else seed + 1)
