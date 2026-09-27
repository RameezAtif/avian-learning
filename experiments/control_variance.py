"""
Control experiment: isolate the variance-matching term in Cao's CHL.

Cao's CHL equation 4 has two terms for W2:
  termA = (y - ŷ)·h        (error-driven)
  termB = γ(Σy² - Σŷ²)/B · W2   (variance-matching)

Our diagnostic showed termB is ~9× larger than termA on this data.
The question: if we take GD, train it, then manually rescale its W2
by sqrt(Var(y) / Var(ŷ)), does the rescaled GD match CHL?

If yes → CHL's advantage is almost entirely the variance term.
If no  → CHL has an advantage beyond variance matching.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.teacher import create_teacher, generate_teacher_experiences
from experiments.config import ExperimentConfig
from learning.gradient_descent import GradientDescentRule
from learning.chl import ContrastiveHebbianRule
from models.student import Student
from memory.notebook import SparseHopfieldNotebook


def build_data(config):
    teacher = create_teacher(config.input_dim, seed=config.seed)
    tutor = generate_teacher_experiences(
        teacher, config.num_tutor_examples,
        1.0 / config.tutor_snr, config.seed + 1, "tutor",
    )
    practice = generate_teacher_experiences(
        teacher, config.num_practice_examples,
        1.0 / config.practice_snr, config.seed + 2, "practice",
    )
    train_x = np.vstack((tutor.x, practice.x))
    train_y = np.concatenate((tutor.y, practice.y))

    evaluation_noise = (
        0.0 if np.isinf(config.evaluation_snr)
        else 1.0 / config.evaluation_snr
    )
    validation = generate_teacher_experiences(
        teacher, config.num_validation_examples,
        evaluation_noise, config.seed + 3, "evaluation",
    )
    return train_x, train_y, validation.x, validation.y


def run_replay(student, notebook, rule, config, epochs):
    for _ in range(epochs):
        results = notebook.replay_stored_batch(config.replays_per_epoch)
        x = np.stack([r.x for r in results])
        y = np.asarray([r.y for r in results])

        out = student.forward(x)
        update = rule.calculate_update(
            x=x, y=y, h=out.h_ff, y_hat=out.y_hat,
            w1=student.W1, w2=student.W2,
        )
        student.W1 += update.delta_w1
        if config.update_w2:
            student.W2 += update.delta_w2


def mse(y, y_hat):
    return 0.5 * float(np.mean((y - y_hat) ** 2))


def main():
    config = ExperimentConfig(seed=42)
    train_x, train_y, val_x, val_y = build_data(config)

    initial_loss = mse(val_y, np.zeros_like(val_y))

    # ---------------- GD ----------------
    gd_student = Student(config.input_dim, config.hidden_dim, seed=config.seed)
    gd_notebook = SparseHopfieldNotebook(
        config.notebook_dim, config.notebook_sparsity, seed=config.seed,
    )
    gd_notebook.encode_batch(x=train_x, y=train_y)

    gd_rule = GradientDescentRule(
        learning_rate=config.learning_rate,
        update_w2=config.update_w2,
        gradient_clip=config.gradient_clip,
    )
    run_replay(gd_student, gd_notebook, gd_rule, config, config.max_epochs)

    gd_out = gd_student.forward(val_x)
    gd_loss = mse(val_y, gd_out.y_hat)

    # ---------------- GD + variance rescale ----------------
    var_y = float(np.var(val_y))
    var_y_hat_gd = float(np.var(gd_out.y_hat))
    scale = np.sqrt(var_y / (var_y_hat_gd + 1e-12))

    gd_scaled = Student(config.input_dim, config.hidden_dim, seed=config.seed)
    gd_scaled.W1 = gd_student.W1.copy()
    gd_scaled.W2 = gd_student.W2 * scale

    gd_scaled_out = gd_scaled.forward(val_x)
    gd_scaled_loss = mse(val_y, gd_scaled_out.y_hat)

    # ---------------- CHL (Cao literal) ----------------
    chl_student = Student(config.input_dim, config.hidden_dim, seed=config.seed)
    chl_notebook = SparseHopfieldNotebook(
        config.notebook_dim, config.notebook_sparsity, seed=config.seed,
    )
    chl_notebook.encode_batch(x=train_x, y=train_y)

    chl_rule = ContrastiveHebbianRule(
        learning_rate=config.learning_rate,
        gamma=1.0,
        eta=0.0,
        gradient_clip=config.gradient_clip,
        update_w2=config.update_w2,
    )
    run_replay(chl_student, chl_notebook, chl_rule, config, config.max_epochs)

    chl_out = chl_student.forward(val_x)
    chl_loss = mse(val_y, chl_out.y_hat)

    # ---------------- Report ----------------
    def rel_improvement(loss):
        return 100 * (1 - loss / initial_loss)

    print()
    print("=" * 68)
    print("CONTROL EXPERIMENT: is CHL just variance-matching?")
    print("=" * 68)
    print(f"Initial validation loss:            {initial_loss:.6f}")
    print()
    print(f"GD final loss:                      {gd_loss:.6f}   "
          f"({rel_improvement(gd_loss):.2f}%)")
    print(f"GD + variance rescale:              {gd_scaled_loss:.6f}   "
          f"({rel_improvement(gd_scaled_loss):.2f}%)")
    print(f"CHL (Cao literal):                  {chl_loss:.6f}   "
          f"({rel_improvement(chl_loss):.2f}%)")
    print()
    print(f"Var(y) on validation:               {var_y:.6f}")
    print(f"Var(y_hat) after GD:                {var_y_hat_gd:.6f}")
    print(f"GD scale factor applied:            {scale:.4f}")
    print()

    gap_before = (gd_loss - chl_loss) / gd_loss * 100
    gap_after = (gd_scaled_loss - chl_loss) / gd_loss * 100

    print(f"GD vs CHL gap (before rescale):     {gap_before:.2f}%")
    print(f"GD vs CHL gap (after rescale):      {gap_after:.2f}%")
    print()

    if abs(gd_scaled_loss - chl_loss) / chl_loss < 0.1:
        print("CONCLUSION: Rescaled GD matches CHL.")
        print("CHL's advantage is almost entirely the variance-matching term.")
    elif gd_scaled_loss < chl_loss * 1.2:
        print("CONCLUSION: Rescaled GD is close to CHL.")
        print("The variance term explains most of CHL's advantage.")
    else:
        print("CONCLUSION: Rescaled GD does not match CHL.")
        print("CHL's advantage comes from something beyond variance matching.")


if __name__ == "__main__":
    main()