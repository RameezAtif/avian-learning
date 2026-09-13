import numpy as np

from data.teacher import generate_teacher_dataset


def test_dataset_shapes():
    dataset = generate_teacher_dataset(
        num_examples=100,
        input_dim=100,
        snr=4,
        seed=42,
    )

    assert dataset.x.shape == (100, 100)
    assert dataset.y.shape == (100,)
    assert dataset.teacher_weights.shape == (100,)
    assert dataset.noise.shape == (100,)


def test_reproducibility():
    dataset_1 = generate_teacher_dataset(
        num_examples=100,
        input_dim=100,
        snr=4,
        seed=42,
    )

    dataset_2 = generate_teacher_dataset(
        num_examples=100,
        input_dim=100,
        snr=4,
        seed=42,
    )

    assert np.array_equal(dataset_1.x, dataset_2.x)
    assert np.array_equal(dataset_1.y, dataset_2.y)
    assert np.array_equal(dataset_1.teacher_weights, dataset_2.teacher_weights)


def test_noiseless_teacher():
    dataset = generate_teacher_dataset(
        num_examples=100,
        input_dim=100,
        snr=np.inf,
        seed=42,
    )

    # With infinite SNR there should be no noise.
    assert np.allclose(dataset.noise, 0.0)


def test_snr_configuration():
    dataset = generate_teacher_dataset(
        num_examples=10000,
        input_dim=100,
        snr=4,
        seed=42,
    )

    # The requested SNR is part of the teacher's theoretical
    # data-generating process.
    assert dataset.snr == 4

    # For SNR = 4:
    #
    # sigma_w^2 = 4 / (4 + 1) = 0.8
    # sigma_epsilon^2 = 1 / (4 + 1) = 0.2
    #
    # The sampled empirical variances do not have to be exactly
    # these values because the teacher and noise are random.
    assert dataset.signal_variance > 0
    assert dataset.noise_variance > 0