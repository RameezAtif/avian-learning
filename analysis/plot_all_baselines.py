"""
Training and validation curves for all rules × all conditions.
X-axis is gradient steps so different regimes are directly comparable.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = PROJECT_ROOT / "results"
OUT = RESULTS / "baseline_plots"
OUT.mkdir(parents=True, exist_ok=True)

RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
LABELS = {
    "gradient_descent": "Gradient Descent",
    "chl": "CHL",
    "qpc": "QPC",
    "hebbian": "Pure Hebbian",
    "anti_hebbian": "Anti-Hebbian",
}

# History file locations + steps per epoch
CONDITIONS = {
    "Online (natural)": {
        "file": RESULTS / "baseline_online" / "baseline_online_histories.csv",
        "steps_per_epoch": 100,
        "color": "#4477AA",
    },
    "CLS": {
        "file": RESULTS / "baseline_cls" / "baseline_cls_histories.csv",
        "steps_per_epoch": 1,
        "color": "#CC6677",
    },
    "Go-CLS": {
        "file": RESULTS / "rq1_baseline" / "rq1_histories.csv",
        "steps_per_epoch": 1,
        "color": "#228833",
    },
}


def load_all():
    frames = []
    for cond, info in CONDITIONS.items():
        path = info["file"]
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        df = pd.read_csv(path)
        df["condition"] = cond
        df["steps"] = df["epoch"] * info["steps_per_epoch"]
        frames.append(df)
    if not frames:
        raise RuntimeError("No history files found.")
    return pd.concat(frames, ignore_index=True)


def smooth(series, window=15):
    return pd.Series(series).rolling(window=window, min_periods=1).mean().values


def plot_curves(df, column, title, filename):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)

    for ax, rule in zip(axes, RULES):
        sub = df[df["learning_rule"] == rule]

        for cond, info in CONDITIONS.items():
            csub = sub[sub["condition"] == cond]
            if csub.empty:
                continue
            agg = csub.groupby("steps")[column].agg(["mean", "std"]).reset_index()
            mean = smooth(agg["mean"].values, window=8)
            std = agg["std"].fillna(0).values

            ax.plot(
                agg["steps"], mean,
                label=cond, color=info["color"], linewidth=1.8,
            )
            ax.fill_between(
                agg["steps"],
                mean - std, mean + std,
                alpha=0.15, color=info["color"],
            )

        ax.set_xscale("log")
        ax.set_xlabel("Gradient steps (log scale)")
        ax.set_title(LABELS[rule])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(column)
    axes[-1].legend(loc="best")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / filename}")


def main():
    df = load_all()
    print(f"Loaded {len(df)} rows, conditions: {sorted(df['condition'].unique())}")

    # Training curves
    plot_curves(
        df, "train_loss",
        "Training loss vs gradient steps",
        "training_curves_all.png",
    )

    # Validation curves
    plot_curves(
        df, "validation_loss",
        "Validation loss vs gradient steps",
        "validation_curves_all.png",
    )


if __name__ == "__main__":
    main()