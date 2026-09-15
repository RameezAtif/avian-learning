"""
Acquisition metrics for the Go-CLS framework.

Acquisition is defined as the first epoch where validation loss reaches
a target level derived from the maximum achievable improvement.

The maximum achievable improvement is determined empirically across all
rules and seeds, so the threshold adapts to the task.
"""

from __future__ import annotations

import numpy as np


def compute_target_loss(
    initial_loss: float,
    best_achievable_loss: float,
    fraction: float = 0.5,
) -> float:
    """
    Target = initial_loss - fraction * (initial_loss - best_achievable)
    """
    return initial_loss - fraction * (initial_loss - best_achievable_loss)


def compute_acquisition_epoch(
    losses: np.ndarray,
    target_loss: float,
    min_consecutive: int = 3,
) -> int | None:
    """
    First epoch where validation loss stays below target for
    `min_consecutive` consecutive epochs. Returns None if never reached.
    """
    count = 0
    for epoch, loss in enumerate(losses):
        if loss <= target_loss:
            count += 1
            if count >= min_consecutive:
                return epoch - min_consecutive + 1
        else:
            count = 0
    return None


def compute_all_acquisition_epochs(
    histories: dict[str, np.ndarray],
    initial_losses: dict[str, float],
    fraction: float = 0.5,
    min_consecutive: int = 3,
) -> dict[str, int | None]:
    """
    histories: {run_key -> array of validation losses}
    initial_losses: {run_key -> initial validation loss}
    """
    best_achievable = min(
        float(np.min(losses)) for losses in histories.values()
    )

    results: dict[str, int | None] = {}
    for key, losses in histories.items():
        target = compute_target_loss(
            initial_loss=initial_losses[key],
            best_achievable_loss=best_achievable,
            fraction=fraction,
        )
        results[key] = compute_acquisition_epoch(
            losses=losses,
            target_loss=target,
            min_consecutive=min_consecutive,
        )
    return results