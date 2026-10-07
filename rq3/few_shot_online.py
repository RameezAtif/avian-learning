"""
RQ3 — Few-shot acquisition experiment (ONLINE / streaming).

Same as rq3/few_shot.py but trains one sample at a time.
No notebook, no replay. Each epoch = one pass through the data,
sample by sample.

Run:
    python rq3/few_shot_online.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rq3.data import generate_tree_dataset
from rq3.models import VectorStudent
from rq3.learning import (
    ContinuousPlasticityVector, ContrastiveHebbianVector,
)


RESULTS = PROJECT_ROOT / "rq3" / "results" / "few_shot_online"
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

BUDGETS = [1, 2, 5, 10, 20]

MAX_EPOCHS = 300
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
GRADIENT_CLIP = 1.0
MAX_W_NORM = 20.0

# Same acquisition criterion as the batch version.
ACQUISITION_FRACTION = 0.5
MIN_CONSECUTIVE = 5


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


def compute_acquisition_epoch(losses, initial_loss,
                              fraction=ACQUISITION_FRACTION,
                              min_consecutive=MIN_CONSECUTIVE):
    """
    First epoch where loss stays below (fraction * initial_loss)
    for at least min_consecutive consecutive epochs.

    Returns the 1-indexed epoch number, or None if never acquired.
    """
    threshold = fraction * initial_loss
    count = 0
    for epoch_idx, loss in enumerate(losses):
        if loss <= threshold:
            count += 1
            if count >= min_consecutive:
                return epoch_idx - min_consecutive + 2
        else:
            count = 0
    return None


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

    student = VectorStudent(
        input_dim=8, hidden_dim=HIDDEN_DIM, output_dim=8, seed=seed,
    )
    rule = make_rule(rule_name)

    # Validation on clean full dataset
    val_x, val_y = X, Y
    initial_loss = float(np.mean((val_y - student.forward(val_x)[1]) ** 2))

    losses = []
    for epoch in range(MAX_EPOCHS):
        # Streaming: shuffle and present one sample at a time
        order = rng.permutation(len(train_x))
        for idx in order:
            x_single = train_x[idx:idx + 1]     # shape (1, 8)
            y_single = train_y[idx:idx + 1]     # shape (1, 8)

            h, y_hat = student.forward(x_single)
            upd = rule.calculate_update(
                x=x_single, y=y_single, h=h, y_hat=y_hat,
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

        # Validate after each pass
        val_loss = float(np.mean((val_y - student.forward(val_x)[1]) ** 2))
        losses.append(val_loss)

    losses = np.array(losses)

    # Final performance: mean of last 10% of epochs
    n_final = max(1, int(0.1 * MAX_EPOCHS))
    final_plateau = float(np.mean(losses[-n_final:]))

    # Acquisition epoch (fixed criterion)
    acquisition_epoch = compute_acquisition_epoch(losses, initial_loss)

    improvement = 100.0 * (1.0 - final_plateau / initial_loss)

    return {
        "learning_rule": rule_name,
        "seed": seed,
        "n_examples_per_object": n_examples_per_object,
        "initial_loss": initial_loss,
        "final_loss": final_plateau,
        "improvement_pct": improvement,
        "acquisition_epoch": acquisition_epoch,
        "losses": losses,
    }


def main():
    seeds = list(range(42, 72))       

    summary_rows = []
    history_rows = []

    total = len(RULES) * len(BUDGETS) * len(seeds)
    count = 0

    for rule in RULES:
        for budget in BUDGETS:
            for seed in seeds:
                count += 1
                print(f"[{count}/{total}] {rule} n={budget} seed={seed}")
                result = train_one(rule, seed, budget)

                summary_rows.append({
                    k: v for k, v in result.items() if k != "losses"
                })

                for epoch_idx, loss in enumerate(result["losses"]):
                    history_rows.append({
                        "learning_rule": rule,
                        "seed": seed,
                        "n_examples_per_object": budget,
                        "epoch": epoch_idx + 1,
                        "val_loss": loss,
                    })

    summary_df = pd.DataFrame(summary_rows)
    history_df = pd.DataFrame(history_rows)

    summary_df.to_csv(RESULTS / "few_shot_online_summary.csv", index=False)
    history_df.to_csv(RESULTS / "few_shot_online_histories.csv", index=False)

    print()
    print("=" * 72)
    print("FEW-SHOT ONLINE — mean improvement per (rule, budget)")
    print("=" * 72)
    pivot_imp = summary_df.pivot_table(
        index="learning_rule", columns="n_examples_per_object",
        values="improvement_pct", aggfunc="mean",
    )
    print(pivot_imp.round(2))

    print()
    print("FEW-SHOT ONLINE — mean acquisition epoch per (rule, budget)")
    print("=" * 72)
    pivot_acq = summary_df.pivot_table(
        index="learning_rule", columns="n_examples_per_object",
        values="acquisition_epoch", aggfunc="mean",
    )
    print(pivot_acq.round(1))

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for rule in RULES:
        sub = summary_df[summary_df["learning_rule"] == rule]
        agg = sub.groupby("n_examples_per_object")["improvement_pct"].mean()
        axes[0].plot(agg.index, agg.values, marker="o",
                     label=LABELS[rule], color=COLORS[rule])
    axes[0].set_xlabel("Examples per object")
    axes[0].set_ylabel("Final improvement (%)")
    axes[0].set_title("Few-shot ONLINE: how well each rule learns")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for rule in RULES:
        sub = summary_df[summary_df["learning_rule"] == rule]
        agg = sub.groupby("n_examples_per_object")["acquisition_epoch"].mean()
        axes[1].plot(agg.index, agg.values, marker="o",
                     label=LABELS[rule], color=COLORS[rule])
    axes[1].set_xlabel("Examples per object")
    axes[1].set_ylabel("Acquisition epoch")
    axes[1].set_title("Few-shot ONLINE: how rapidly each rule acquires")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "few_shot_online_results.png", dpi=300)
    plt.close(fig)

    print()
    print(f"Saved: {RESULTS / 'few_shot_online_summary.csv'}")
    print(f"Saved: {RESULTS / 'few_shot_online_histories.csv'}")
    print(f"Saved: {RESULTS / 'few_shot_online_results.png'}")


if __name__ == "__main__":
    main()