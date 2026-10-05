"""
RQ3 — Few-shot acquisition experiment.

Varies the number of training examples per object and measures
how quickly each learning rule acquires the tree structure.

Answers the "rapid acquisition from short, repetitive sequences"
sub-question of RQ3.

Run:
    python rq3/few_shot.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rq3.data import generate_tree_dataset, SimpleNotebook
from rq3.models import VectorStudent
from rq3.learning import (
    ContinuousPlasticityVector, ContrastiveHebbianVector,
)
from rq3.metrics import categorical_boundary_strength


RESULTS = PROJECT_ROOT / "rq3" / "results" / "few_shot"
RESULTS.mkdir(parents=True, exist_ok=True)

RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
LABELS = {
    "gradient_descent": "GD",
    "chl": "CHL",
    "qpc": "QPC",
    "hebbian": "Hebbian",
    "anti_hebbian": "Anti-Hebbian",
}
COLORS = {
    "gradient_descent": "tab:blue",
    "chl": "tab:orange",
    "qpc": "tab:green",
    "hebbian": "tab:red",
    "anti_hebbian": "tab:purple",
}

# How many examples per object to try
BUDGETS = [1, 2, 5, 10, 20]

# Training settings
MAX_EPOCHS = 300
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
GRADIENT_CLIP = 1.0
MAX_W_NORM = 20.0

# Acquisition: first epoch where loss is within 50% of final plateau.
# "Final plateau" = mean loss over the last 10% of epochs.
ACQUISITION_FRACTION = 0.5


def make_rule(name):
    if name == "gradient_descent":
        return ContinuousPlasticityVector(1.0, 0.0, LEARNING_RATE, GRADIENT_CLIP)
    if name == "chl":
        return ContrastiveHebbianVector(
            LEARNING_RATE, gamma=1.0, eta=0.0, gradient_clip=GRADIENT_CLIP,
        )
    if name == "qpc":
        return ContinuousPlasticityVector(-1.0, 0.0, LEARNING_RATE, GRADIENT_CLIP)
    if name == "hebbian":
        return ContinuousPlasticityVector(0.0, 0.1, LEARNING_RATE, GRADIENT_CLIP)
    if name == "anti_hebbian":
        return ContinuousPlasticityVector(0.0, -0.1, LEARNING_RATE, GRADIENT_CLIP)
    raise ValueError(name)


def train_one(rule_name, seed, n_examples_per_object):
    rng = np.random.default_rng(seed)
    X, Y = generate_tree_dataset()

    # Build limited training set: n_examples_per_object copies per object
    train_x, train_y = [], []
    for obj_idx in range(8):
        for _ in range(n_examples_per_object):
            train_x.append(X[obj_idx] + rng.normal(0.0, 0.3, 8))
            train_y.append(Y[obj_idx])
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    notebook = SimpleNotebook()
    notebook.encode(train_x, train_y)

    student = VectorStudent(
        input_dim=8, hidden_dim=HIDDEN_DIM, output_dim=8, seed=seed,
    )
    rule = make_rule(rule_name)

    # Validation on clean full dataset
    val_x, val_y = X, Y
    initial_loss = float(np.mean((val_y - student.forward(val_x)[1]) ** 2))

    # Batch size: one full pass through the small dataset
    batch_size = min(16, len(train_x))

    losses = []
    for epoch in range(MAX_EPOCHS):
        xb, yb = notebook.sample(batch_size, rng)
        h, y_hat = student.forward(xb)
        upd = rule.calculate_update(
            x=xb, y=yb, h=h, y_hat=y_hat,
            w1=student.W1, w2=student.W2,
        )
        student.W1 += upd.delta_w1
        student.W2 += upd.delta_w2

        # Safety clip
        w1n = np.linalg.norm(student.W1, "fro")
        if w1n > MAX_W_NORM:
            student.W1 *= MAX_W_NORM / w1n
        w2n = np.linalg.norm(student.W2, "fro")
        if w2n > MAX_W_NORM:
            student.W2 *= MAX_W_NORM / w2n

        val_loss = float(np.mean((val_y - student.forward(val_x)[1]) ** 2))
        losses.append(val_loss)

    losses = np.array(losses)

    # Final plateau: mean of last 10% of epochs
    n_final = max(1, int(0.1 * MAX_EPOCHS))
    final_plateau = float(np.mean(losses[-n_final:]))

    # Acquisition: first epoch where loss reaches within 50% of the gap
    # between initial and plateau. Matches the "best_achievable" criterion
    # used elsewhere in the project.
    threshold = final_plateau + ACQUISITION_FRACTION * (initial_loss - final_plateau)
    below = np.where(losses <= threshold)[0]
    acquisition_epoch = int(below[0] + 1) if len(below) > 0 else None

    improvement = 100.0 * (1.0 - final_plateau / initial_loss)

    return {
        "learning_rule": rule_name,
        "seed": seed,
        "n_examples_per_object": n_examples_per_object,
        "initial_loss": initial_loss,
        "final_loss": final_plateau,
        "improvement_pct": improvement,
        "acquisition_epoch": acquisition_epoch,
    }


def main():
    seeds = list(range(42, 52))       # 10 seeds

    rows = []
    total = len(RULES) * len(BUDGETS) * len(seeds)
    count = 0

    for rule in RULES:
        for budget in BUDGETS:
            for seed in seeds:
                count += 1
                print(f"[{count}/{total}] {rule} n={budget} seed={seed}")
                result = train_one(rule, seed, budget)
                rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "few_shot_summary.csv", index=False)

    # ---------------- Print summary ----------------
    print()
    print("=" * 72)
    print("FEW-SHOT ACQUISITION — mean improvement per (rule, budget)")
    print("=" * 72)
    pivot_imp = df.pivot_table(
        index="learning_rule", columns="n_examples_per_object",
        values="improvement_pct", aggfunc="mean",
    )
    print(pivot_imp.round(2))
    print()
    print("FEW-SHOT ACQUISITION — mean acquisition epoch per (rule, budget)")
    print("=" * 72)
    pivot_acq = df.pivot_table(
        index="learning_rule", columns="n_examples_per_object",
        values="acquisition_epoch", aggfunc="mean",
    )
    print(pivot_acq.round(1))

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Improvement vs budget
    for rule in RULES:
        sub = df[df["learning_rule"] == rule]
        agg = sub.groupby("n_examples_per_object")["improvement_pct"].mean()
        axes[0].plot(agg.index, agg.values, marker="o",
                     label=LABELS[rule], color=COLORS[rule])
    axes[0].set_xlabel("Examples per object")
    axes[0].set_ylabel("Final improvement (%)")
    axes[0].set_title("Few-shot: how well each rule learns")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Acquisition epoch vs budget
    for rule in RULES:
        sub = df[df["learning_rule"] == rule]
        agg = sub.groupby("n_examples_per_object")["acquisition_epoch"].mean()
        axes[1].plot(agg.index, agg.values, marker="o",
                     label=LABELS[rule], color=COLORS[rule])
    axes[1].set_xlabel("Examples per object")
    axes[1].set_ylabel("Acquisition epoch")
    axes[1].set_title("Few-shot: how rapidly each rule acquires")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "few_shot_results.png", dpi=300)
    plt.close(fig)

    print()
    print(f"Saved: {RESULTS / 'few_shot_summary.csv'}")
    print(f"Saved: {RESULTS / 'few_shot_results.png'}")


if __name__ == "__main__":
    main()