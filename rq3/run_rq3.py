"""
RQ3: hierarchical tree learning with five rules.

Run from project root:
    python rq3/run_rq3.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from rq3.data import (
    generate_tree_dataset, add_input_noise,
    SimpleNotebook, CATEGORY_LABELS,
)
from rq3.models import VectorStudent
from rq3.learning import (
    ContinuousPlasticityVector, ContrastiveHebbianVector,
)
from rq3.metrics import (
    categorical_boundary_strength, progressive_differentiation,
)

RESULTS = PROJECT_ROOT / "rq3" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def make_rule(name, lr=0.01, clip=1.0):
    if name == "gradient_descent":
        return ContinuousPlasticityVector(1.0, 0.0, lr, clip)
    if name == "chl":
        return ContrastiveHebbianVector(lr, gamma=1.0, eta=0.0,
                                        gradient_clip=clip)
    if name == "qpc":
        return ContinuousPlasticityVector(-1.0, 0.0, lr, clip)
    if name == "hebbian":
        return ContinuousPlasticityVector(0.0, 0.1, lr, clip)
    if name == "anti_hebbian":
        return ContinuousPlasticityVector(0.0, -0.1, lr, clip)
    raise ValueError(name)


def train_one(rule_name, seed, max_epochs=300, batch_size=16,
              tutor_snr=20.0, practice_snr=3.0, hidden_dim=32):

    rng = np.random.default_rng(seed)
    X, Y = generate_tree_dataset()

    # Build tutor (low noise) and practice (high noise) from same data
    tutor_x, tutor_y = add_input_noise(X, Y, 1.0 / np.sqrt(tutor_snr), rng)
    practice_x, practice_y = add_input_noise(X, Y, 1.0 / np.sqrt(practice_snr), rng)

    train_x = np.vstack([tutor_x, practice_x])
    train_y = np.vstack([tutor_y, practice_y])

    notebook = SimpleNotebook()
    notebook.encode(train_x, train_y)

    student = VectorStudent(
        input_dim=8, hidden_dim=hidden_dim, output_dim=8, seed=seed,
    )
    rule = make_rule(rule_name)

    # validation: clean data
    val_x, val_y = X, Y

    initial_loss = float(np.mean((val_y - student.forward(val_x)[1]) ** 2))

    history = []
    epoch = 0
    # Cap on weights to avoid runaway on hierarchical CHL
    MAX_W_NORM = 20.0

    while epoch < max_epochs:
        epoch += 1
        xb, yb = notebook.sample(batch_size, rng)
        h, y_hat = student.forward(xb)

        update = rule.calculate_update(
            x=xb, y=yb, h=h, y_hat=y_hat,
            w1=student.W1, w2=student.W2,
        )
        student.W1 += update.delta_w1
        student.W2 += update.delta_w2

        # Safety net: clip weight norms
        w1n = np.linalg.norm(student.W1, "fro")
        if w1n > MAX_W_NORM:
            student.W1 *= MAX_W_NORM / w1n
        w2n = np.linalg.norm(student.W2, "fro")
        if w2n > MAX_W_NORM:
            student.W2 *= MAX_W_NORM / w2n

        val_loss = float(np.mean(
            (val_y - student.forward(val_x)[1]) ** 2
        ))

        # boundary strength on level2 category (Bird/Fish/Tree/Flower)
        _, h_val = student.forward(val_x)
        boundary = categorical_boundary_strength(
            h_val, CATEGORY_LABELS["level2"]
        )

        # progressive differentiation: singular values
        sv = progressive_differentiation(student, val_x, val_y)

        row = {
            "epoch": epoch,
            "val_loss": val_loss,
            "boundary_strength": boundary,
        }
        # Log up to 8 singular values; pad with NaN if fewer exist
        for i in range(8):
            row[f"sv_{i}"] = float(sv[i]) if i < len(sv) else float("nan")
        history.append(row)

    final_loss = history[-1]["val_loss"]
    improvement = 100.0 * (1 - final_loss / initial_loss)

    return {
        "rule": rule_name,
        "seed": seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement_pct": improvement,
        "history": history,
    }


def main():
    rules = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
    seeds = list(range(42, 52))     # 10 seeds

    summary_rows = []
    history_rows = []

    total = len(rules) * len(seeds)
    count = 0

    for rule in rules:
        for seed in seeds:
            count += 1
            print(f"[{count}/{total}] {rule} seed={seed}")
            result = train_one(rule, seed)

            summary_rows.append({
                "learning_rule": result["rule"],
                "seed": result["seed"],
                "initial_loss": result["initial_loss"],
                "final_loss": result["final_loss"],
                "improvement_pct": result["improvement_pct"],
            })

            for row in result["history"]:
                history_rows.append({
                    "learning_rule": rule,
                    "seed": seed,
                    **row,
                })

    pd.DataFrame(summary_rows).to_csv(
        RESULTS / "rq3_summary.csv", index=False,
    )
    pd.DataFrame(history_rows).to_csv(
        RESULTS / "rq3_histories.csv", index=False,
    )

    print()
    print("=" * 68)
    print("RQ3 SUMMARY")
    print("=" * 68)
    print(pd.DataFrame(summary_rows).groupby(
        "learning_rule")["improvement_pct"].agg(["mean", "std", "count"]))
    print()
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()