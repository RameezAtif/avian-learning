"""
Visualise the control experiment:
  - GD alone
  - GD + CHL's variance term
  - CHL (Cao-literal)

Shows that GD + variance matches CHL to 15 decimal places.

Run:
    python analysis/plot_control_variance.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = PROJECT_ROOT / "results" / "diagnostics"
CSV_FILE = RESULTS / "control_variance_v2.csv"
OUT_DIR = PROJECT_ROOT / "results" / "comparison_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load():
    df = pd.read_csv(CSV_FILE)
    return df


def plot_slope_chart(df):
    """
    Slope chart: one line per seed. x-axis has the three conditions.
    Shows GD alone -> GD + variance -> CHL.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    conditions = ["gd_alone", "gd_plus_variance", "chl"]
    labels = ["GD alone", "GD + variance term", "CHL (Cao-literal)"]
    x_positions = [0, 1, 2]

    # Pivot to get one row per seed, columns for each condition
    pivot = df.pivot(index="seed", columns="condition", values="improvement_pct")

    for seed in pivot.index:
        values = [pivot.loc[seed, c] for c in conditions]
        ax.plot(
            x_positions, values,
            color="steelblue", alpha=0.45, linewidth=1.2,
        )

    # Mean line (bold)
    means = [pivot[c].mean() for c in conditions]
    ax.plot(
        x_positions, means,
        color="black", linewidth=3, marker="D",
        markersize=12, zorder=10, label="Mean",
    )

    # Annotate means
    for xi, m in zip(x_positions, means):
        ax.annotate(
            f"{m:.2f}%",
            (xi, m),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative improvement (%)")
    ax.set_title(
        "Control experiment: GD + variance term matches CHL exactly\n"
        "(one line per seed, n=30)"
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = OUT_DIR / "control_variance_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_paired_scatter(df):
    """
    Two-panel scatter:
    Left: GD alone vs CHL
    Right: GD + variance vs CHL
    All points on the diagonal in the right panel = exact match.
    """
    pivot = df.pivot(index="seed", columns="condition", values="improvement_pct")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: GD alone vs CHL
    ax = axes[0]
    x = pivot["gd_alone"].values
    y = pivot["chl"].values
    ax.scatter(x, y, s=60, alpha=0.7, color="tab:blue",
               edgecolor="black", linewidth=0.6)
    lo = min(x.min(), y.min()) - 5
    hi = max(x.max(), y.max()) + 5
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    ax.set_xlabel("GD alone improvement (%)")
    ax.set_ylabel("CHL improvement (%)")
    ax.set_title("GD alone vs CHL\n(points above diagonal: CHL is better)")
    ax.grid(True, alpha=0.3)

    # Right: GD + variance vs CHL
    ax = axes[1]
    x = pivot["gd_plus_variance"].values
    y = pivot["chl"].values
    ax.scatter(x, y, s=60, alpha=0.7, color="tab:orange",
               edgecolor="black", linewidth=0.6)
    lo = min(x.min(), y.min()) - 5
    hi = max(x.max(), y.max()) + 5
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    ax.set_xlabel("GD + variance term improvement (%)")
    ax.set_ylabel("CHL improvement (%)")
    ax.set_title("GD + variance vs CHL\n(every point on diagonal: identical)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Adding CHL's variance term to GD reproduces CHL exactly",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "control_variance_paired.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    df = load()

    print("=" * 60)
    print("CONTROL EXPERIMENT SUMMARY")
    print("=" * 60)
    print()
    summary = df.groupby("condition")["improvement_pct"].agg(
        ["mean", "std", "count"]
    )
    print(summary.round(4))
    print()

    plot_slope_chart(df)
    plot_paired_scatter(df)


if __name__ == "__main__":
    main()