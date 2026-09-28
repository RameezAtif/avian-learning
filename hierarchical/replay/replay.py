"""
Replay loop for hierarchical RQ1.

For each epoch:
    1. Sample batch from notebook
    2. Forward through student
    3. Compute MSE loss (pre-update)
    4. Apply learning rule
    5. Record loss
"""

import numpy as np


def mse_loss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean squared error averaged over batch and output dims."""
    return float(np.mean((y - y_hat) ** 2))


def train_student(
    student,
    notebook,
    learning_rule,
    max_epochs: int,
    batch_size: int,
    seed: int | None = None,
):
    """
    Train the student for max_epochs. Returns a history dict.
    """
    rng = np.random.default_rng(seed)
    replay_losses = []

    for epoch in range(max_epochs):
        x, y = notebook.sample_batch(batch_size, rng)
        out = student.forward(x)

        loss = mse_loss(y, out.y_hat)
        replay_losses.append(loss)

        update = learning_rule.calculate_update(
            x=x, y=y, h=out.h, y_hat=out.y_hat,
            w1=student.W1, w2=student.W2,
        )

        student.W1 += update.delta_w1
        student.W2 += update.delta_w2

    return {
        "replay_losses": np.array(replay_losses),
    }


def evaluate_validation(student, val_x, val_y):
    out = student.forward(val_x)
    return mse_loss(val_y, out.y_hat)