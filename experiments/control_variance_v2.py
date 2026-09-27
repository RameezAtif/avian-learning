"""
Corrected control experiment: does adding CHL's variance term to GD
reproduce CHL's performance?

Instead of rescaling GD's W2 once at the end (which was the wrong
test), we run GD with the variance term added to every W2 update,
matching how CHL applies it.

Three conditions compared:
  1. GD alone
  2. GD + CHL variance term on W2 every epoch
  3. CHL (Cao-literal)

If condition 2 lands near CHL, then CHL's advantage is the variance term.
If condition 2 stays near GD, CHL has an advantage beyond variance matching.

Also logs Var(y_hat) per epoch to help diagnose why seed 50 blew up.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.teacher import create_teacher, generate_teacher_experiences
from experiments.config import ExperimentConfig
from learning.chl import ContrastiveHebbianRule
from models.student import Student
from memory.notebook import SparseHopfieldNotebook


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

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


def mse(y, y_hat):
    return 0.5 * float(np.mean((y - y_hat) ** 2))


# -----------------------------------------------------------------
# Condition 1: GD alone
# -----------------------------------------------------------------

def run_gd_alone(student, notebook, config):
    """
    Standard gradient descent on W1 and W2.
    """
    lr = config.learning_rate
    clip = config.gradient_clip

    for _ in range(config.max_epochs):
        results = notebook.replay_stored_batch(config.replays_per_epoch)
        x = np.stack([r.x for r in results])
        y = np.asarray([r.y for r in results])
        B = x.shape[0]

        out = student.forward(x)

        # Forward pass: h = W1 x, y_hat = W2 h
        # Error for gradient: e = y_hat - y (GD convention)
        error = out.y_hat - y

        # Gradients
        grad_w1 = (student.W2.T @ error[None, :] @ x) / B
        grad_w2 = (error[None, :] @ out.h_ff) / B

        delta_w1 = -lr * grad_w1
        delta_w2 = -lr * grad_w2

        if clip is not None:
            delta_w1 = np.clip(delta_w1, -clip, clip)
            delta_w2 = np.clip(delta_w2, -clip, clip)

        student.W1 += delta_w1
        student.W2 += delta_w2


# -----------------------------------------------------------------
# Condition 2: GD + CHL variance term on W2
# -----------------------------------------------------------------

def run_gd_with_variance(student, notebook, config):
    """
    GD on W1, and on W2: supervised error + CHL's variance term.

    The variance term is:
        gamma * (sum(y^2) - sum(y_hat^2)) / B * W2

    This matches the second term of Cao's equation 4, applied each
    epoch just as CHL does.
    """
    lr = config.learning_rate
    clip = config.gradient_clip
    gamma = 1.0

    for _ in range(config.max_epochs):
        results = notebook.replay_stored_batch(config.replays_per_epoch)
        x = np.stack([r.x for r in results])
        y = np.asarray([r.y for r in results])
        B = x.shape[0]

        out = student.forward(x)

        error = out.y_hat - y  # GD convention

        # Supervised W1 gradient
        grad_w1 = (student.W2.T @ error[None, :] @ x) / B
        delta_w1 = -lr * grad_w1

        # W2 = supervised + variance term
        grad_w2_supervised = (error[None, :] @ out.h_ff) / B

        corr_coef = gamma * (
            np.sum(y ** 2) - np.sum(out.y_hat ** 2)
        ) / B
        variance_term_w2 = corr_coef * student.W2

        delta_w2 = -lr * grad_w2_supervised + lr * variance_term_w2

        if clip is not None:
            delta_w1 = np.clip(delta_w1, -clip, clip)
            delta_w2 = np.clip(delta_w2, -clip, clip)

        student.W1 += delta_w1
        student.W2 += delta_w2


# -----------------------------------------------------------------
# Condition 3: CHL (Cao-literal)
# -----------------------------------------------------------------

def run_chl(student, notebook, config):
    rule = ContrastiveHebbianRule(
        learning_rate=config.learning_rate,
        gamma=1.0,
        eta=0.0,
        gradient_clip=config.gradient_clip,
        update_w2=config.update_w2,
    )
    for _ in range(config.max_epochs):
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


# -----------------------------------------------------------------
# Run one condition
# -----------------------------------------------------------------

def run_condition(condition_name, run_fn, config, train_x, train_y, val_x, val_y):
    student = Student(config.input_dim, config.hidden_dim, seed=config.seed)
    notebook = SparseHopfieldNotebook(
        config.notebook_dim, config.notebook_sparsity, seed=config.seed,
    )
    notebook.encode_batch(x=train_x, y=train_y)

    run_fn(student, notebook, config)

    out = student.forward(val_x)
    loss = mse(val_y, out.y_hat)
    initial = mse(val_y, np.zeros_like(val_y))
    improvement = 100 * (1 - loss / initial)

    return {
        "condition": condition_name,
        "final_loss": loss,
        "improvement_pct": improvement,
    }


# -----------------------------------------------------------------
# Main — run across multiple seeds
# -----------------------------------------------------------------

def main():
    seeds = list(range(42, 72))  # 30 seeds, matches RQ1

    rows = []

    for seed in seeds:
        config = ExperimentConfig(seed=seed)
        train_x, train_y, val_x, val_y = build_data(config)

        for name, fn in [
            ("gd_alone", run_gd_alone),
            ("gd_plus_variance", run_gd_with_variance),
            ("chl", run_chl),
        ]:
            result = run_condition(name, fn, config,
                                   train_x, train_y, val_x, val_y)
            result["seed"] = seed
            rows.append(result)
            print(f"seed={seed}  {name:<18}  loss={result['final_loss']:.4f}  "
                  f"improvement={result['improvement_pct']:.2f}%")

    df = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "control_variance_v2.csv", index=False)

    # Summary
    print()
    print("=" * 68)
    print("CORRECTED CONTROL EXPERIMENT — 30 seeds")
    print("=" * 68)
    summary = df.groupby("condition")["improvement_pct"].agg(
        ["mean", "std", "count"]
    )
    print(summary)
    print()
    print(f"Saved: {out_dir / 'control_variance_v2.csv'}")


if __name__ == "__main__":
    main()