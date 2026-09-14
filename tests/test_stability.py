import numpy as np
import pytest

from metrics.stability import (
    calculate_windowed_stability,
)


def test_constant_loss_has_zero_stability_variance():
    loss_history = np.ones(100)

    result = calculate_windowed_stability(
        loss_history=loss_history,
        final_fraction=0.10,
        num_windows=5,
    )

    assert result.stability_variance == 0.0


def test_stability_result_window_count():
    loss_history = np.arange(
        1,
        101,
        dtype=float,
    )

    result = calculate_windowed_stability(
        loss_history=loss_history,
        final_fraction=0.10,
        num_windows=5,
    )

    assert result.num_windows == 5
    assert result.window_means.shape == (5,)
    assert result.window_size == 2


def test_window_means_are_correct():
    loss_history = np.arange(
        1,
        101,
        dtype=float,
    )

    result = calculate_windowed_stability(
        loss_history=loss_history,
        final_fraction=0.10,
        num_windows=5,
    )

    # Final 10%:
    #
    # 91 ... 100
    #
    # Windows of size 2:
    #
    # [91, 92]
    # [93, 94]
    # [95, 96]
    # [97, 98]
    # [99, 100]
    #
    # Means:
    #
    # 91.5, 93.5, 95.5, 97.5, 99.5

    expected = np.array(
        [91.5, 93.5, 95.5, 97.5, 99.5]
    )

    assert np.allclose(
        result.window_means,
        expected,
    )


def test_stability_variance_matches_manual_calculation():
    loss_history = np.arange(
        1,
        101,
        dtype=float,
    )

    result = calculate_windowed_stability(
        loss_history=loss_history,
        final_fraction=0.10,
        num_windows=5,
    )

    window_means = np.array(
        [91.5, 93.5, 95.5, 97.5, 99.5]
    )

    grand_mean = np.mean(
        window_means
    )

    expected = np.mean(
        (window_means - grand_mean) ** 2
    )

    assert np.isclose(
        result.stability_variance,
        expected,
    )


def test_decreasing_loss_can_have_nonzero_stability():
    loss_history = np.linspace(
        10.0,
        1.0,
        100,
    )

    result = calculate_windowed_stability(
        loss_history=loss_history,
        final_fraction=0.10,
        num_windows=5,
    )

    assert result.stability_variance > 0.0


def test_invalid_fraction():
    loss_history = np.ones(100)

    with pytest.raises(ValueError):
        calculate_windowed_stability(
            loss_history=loss_history,
            final_fraction=0.0,
            num_windows=5,
        )


def test_invalid_window_count():
    loss_history = np.ones(100)

    with pytest.raises(ValueError):
        calculate_windowed_stability(
            loss_history=loss_history,
            final_fraction=0.10,
            num_windows=0,
        )


def test_empty_loss_history():
    with pytest.raises(ValueError):
        calculate_windowed_stability(
            loss_history=np.array([]),
            final_fraction=0.10,
            num_windows=5,
        )


def test_insufficient_observations():
    loss_history = np.ones(10)

    with pytest.raises(ValueError):
        calculate_windowed_stability(
            loss_history=loss_history,
            final_fraction=0.10,
            num_windows=5,
        )