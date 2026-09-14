import numpy as np

from metrics.stability import (
    calculate_windowed_stability,
)


# ---------------------------------------------------------
# Example loss history
#
# We construct a history that gradually decreases and then
# becomes comparatively stable near the end.
# ---------------------------------------------------------

rng = np.random.default_rng(42)

early_loss = np.linspace(
    5.0,
    1.0,
    900,
)

late_loss = (
    1.0
    + rng.normal(
        loc=0.0,
        scale=0.02,
        size=100,
    )
)

loss_history = np.concatenate(
    [
        early_loss,
        late_loss,
    ]
)


# ---------------------------------------------------------
# Calculate stability
# ---------------------------------------------------------

result = calculate_windowed_stability(
    loss_history=loss_history,
    final_fraction=0.10,
    num_windows=5,
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("===== WINDOWED STABILITY =====")

print(
    "Total loss observations:",
    loss_history.size,
)

print(
    "Final segment size:",
    result.final_segment.size,
)

print(
    "Start index:",
    result.start_index,
)

print(
    "Number of windows:",
    result.num_windows,
)

print(
    "Window size:",
    result.window_size,
)

print()

print(
    "Window means:",
    result.window_means,
)

print(
    "Stability variance S:",
    result.stability_variance,
)