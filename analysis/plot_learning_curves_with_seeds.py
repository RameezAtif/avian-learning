"""
Learning curves for RQ1 with per-seed detail.

Generates two figures:
  1. validation_learning_curves.png — mean ± SD across seeds (the old one)
  2. validation_learning_curves_per_seed.png — small multiples, one panel
     per rule, one line per seed. Shows why the SD is high.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = PROJECT_ROOT / "results" / "rq1_baseline"
HISTORIES = RESULTS / "rq1_histories.csv"
OUT_DIR = PROJECT_ROOT / "results" / "comparison_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
LABELS = {
    "gradient_descent": "Gradient Descent",
    "chl": "CHL",
    "qpc": "QPC",
    "hebbian": "Pure Hebbian",
    "anti_hebbian": "Anti-Hebbian",
}
COLORS = {
    "gradient_descent": "tab:blue",
    "chl": "tab:orange",
    "qpc": "tab:green",
    "hebbian": "tab:red",
    "anti_hebbian": "tab:purple",
}


def load():
    df = pd.read_csv(HISTORIES)
    return df


def plot_aggregated(df):
    """The current figure: mean ± SD across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for rule in RULES:
        sub = df[df["learning_rule"] == rule]
        agg = sub.groupby("epoch")["validation_loss"].agg(
            ["mean", "std"]
        ).reset_index()

        ax.plot(agg["epoch"], agg["mean"],
                label=LABELS[rule], color=COLORS[rule], linewidth=2)
        ax.fill_between(
            agg["epoch"],
            agg["mean"] - agg["std"].fillna(0),
            agg["mean"] + agg["std"].fillna(0),
            alpha=0.18, color=COLORS[rule],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation learning curves — mean ± SD across 30 seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "validation_learning_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_per_seed(df):
    """
    Small multiples: one panel per rule, one line per seed.
    Makes it obvious which seeds are outliers.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    for ax, rule in zip(axes, RULES):
        sub = df[df["learning_rule"] == rule]

        # Plot each seed as a thin line
        for seed, group in sub.groupby("seed"):
            group = group.sort_values("epoch")
            ax.plot(
                group["epoch"], group["validation_loss"],
                color=COLORS[rule], alpha=0.45, linewidth=1.1,
            )

        # Highlight seed 50 and seed 61 (the outliers) if present
        for outlier_seed in [50, 61]:
            outlier = sub[sub["seed"] == outlier_seed].sort_values("epoch")
            if len(outlier) > 0:
                ax.plot(
                    outlier["epoch"], outlier["validation_loss"],
                    color="black", linewidth=2, linestyle="--",
                    label=f"seed {outlier_seed}",
                )

        # Mean line (bold)
        mean_curve = sub.groupby("epoch")["validation_loss"].mean().reset_index()
        ax.plot(
            mean_curve["epoch"], mean_curve["validation_loss"],
            color="black", linewidth=2.5, label="mean",
        )

        ax.set_title(LABELS[rule])
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Validation loss")
    fig.suptitle(
        "Validation learning curves — per seed\n"
        "Most seeds cluster tightly; black dashed lines highlight outliers",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "validation_learning_curves_per_seed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_variance_diagnosis(df):
    """Print per-seed final loss per rule to identify outliers."""
    print()
    print("=" * 72)
    print("PER-SEED FINAL VALIDATION LOSS (last epoch)")
    print("=" * 72)
    print()

    for rule in RULES:
        sub = df[df["learning_rule"] == rule]
        last_epoch = sub["epoch"].max()
        finals = sub[sub["epoch"] == last_epoch].sort_values("validation_loss")

        print(f"\n{LABELS[rule]}:")
        print(f"  min = {finals['validation_loss'].min():.4f} "
              f"(seed {finals.iloc[0]['seed']})")
        print(f"  max = {finals['validation_loss'].max():.4f} "
              f"(seed {finals.iloc[-1]['seed']})")
        print(f"  mean = {finals['validation_loss'].mean():.4f} "
              f"± {finals['validation_loss'].std(ddof=1):.4f}")

        # Show top 3 worst seeds
        worst = finals.tail(3)
        print(f"  worst 3 seeds: " + ", ".join(
            f"seed {int(r['seed'])} ({r['validation_loss']:.3f})"
            for _, r in worst.iterrows()
        ))


def main():
    df = load()

    plot_aggregated(df)
    plot_per_seed(df)
    print_variance_diagnosis(df)


if __name__ == "__main__":
    main()