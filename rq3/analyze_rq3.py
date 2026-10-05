"""
Analysis and figures for RQ3.

Run after run_rq3.py:
    python rq3/analyze_rq3.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = PROJECT_ROOT / "rq3" / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

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
    summary = pd.read_csv(RESULTS / "rq3_summary.csv")
    histories = pd.read_csv(RESULTS / "rq3_histories.csv")
    return summary, histories


def plot_learning_curves(histories):
    fig, ax = plt.subplots(figsize=(9, 5))
    for rule in RULES:
        data = histories[histories["learning_rule"] == rule]
        agg = data.groupby("epoch")["val_loss"].agg(["mean", "std"]).reset_index()
        ax.plot(agg["epoch"], agg["mean"], label=LABELS[rule],
                color=COLORS[rule])
        ax.fill_between(agg["epoch"],
                        agg["mean"] - agg["std"].fillna(0),
                        agg["mean"] + agg["std"].fillna(0),
                        alpha=0.15, color=COLORS[rule])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (log scale)")
    ax.set_yscale("log")
    ax.set_title("RQ3: Validation loss trajectories on hierarchical tree")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_learning_curves.png", dpi=300)
    plt.close(fig)
    print("Saved rq3_learning_curves.png")


def plot_boundary_strength(histories):
    fig, ax = plt.subplots(figsize=(9, 5))
    for rule in RULES:
        data = histories[histories["learning_rule"] == rule]
        agg = data.groupby("epoch")["boundary_strength"].mean().reset_index()
        ax.plot(agg["epoch"], agg["boundary_strength"],
                label=LABELS[rule], color=COLORS[rule])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Boundary strength (between/within)")
    ax.set_title("RQ3: Categorical boundary sharpening over training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_boundary_strength.png", dpi=300)
    plt.close(fig)
    print("Saved rq3_boundary_strength.png")


def plot_progressive_differentiation(histories):
    """Singular values over time, one panel per rule."""
    sv_cols = [f"sv_{i}" for i in range(8)]

    # Only plot columns that actually exist in the data
    present_cols = [c for c in sv_cols if c in histories.columns]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

    for ax, rule in zip(axes, RULES):
        data = histories[histories["learning_rule"] == rule]
        agg = data.groupby("epoch")[present_cols].mean().reset_index()
        for col in present_cols:
            ax.plot(agg["epoch"], agg[col], alpha=0.7)
        ax.set_title(LABELS[rule])
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Singular value")
    fig.suptitle("RQ3: Progressive differentiation (singular values over training)")
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_progressive_differentiation.png", dpi=300)
    plt.close(fig)
    print("Saved rq3_progressive_differentiation.png")


def plot_improvement_bars(summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for rule in RULES:
        vals = summary[summary["learning_rule"] == rule]["improvement_pct"]
        data.append(vals.values)
        labels.append(LABELS[rule])
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Improvement (%)")
    ax.set_title("RQ3: Improvement per rule (10 seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_improvement.png", dpi=300)
    plt.close(fig)
    print("Saved rq3_improvement.png")


def main():
    summary, histories = load()

    print("=" * 68)
    print("RQ3 SUMMARY")
    print("=" * 68)
    print(summary.groupby("learning_rule")["improvement_pct"]
          .agg(["mean", "std", "count"]))
    print()

    plot_learning_curves(histories)
    plot_boundary_strength(histories)
    plot_progressive_differentiation(histories)
    plot_improvement_bars(summary)


if __name__ == "__main__":
    main()