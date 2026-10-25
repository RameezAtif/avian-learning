"""
Training and validation curves for all rules × all conditions.
Layout: 2 rows x 5 columns.
Top row: Training Loss. Bottom row: Validation Loss.
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

CONDITIONS = {
    "Online": {
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
        
        if "steps" in df.columns:
            df["steps"] = df["steps"]
        elif "step" in df.columns:
            df["steps"] = df["step"]
        elif "epoch" in df.columns:
            df["steps"] = df["epoch"] * info["steps_per_epoch"]
            
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def smooth(series, window=5):
    return pd.Series(series).rolling(window=window, min_periods=1).mean().values

def plot_metric(df, column, row_idx, y_limit):
    for col_idx, rule in enumerate(RULES):
        ax = axes[row_idx, col_idx]
        sub = df[df["learning_rule"] == rule]

        for cond, info in CONDITIONS.items():
            csub = sub[sub["condition"] == cond]
            if csub.empty:
                continue
            
            agg = csub.groupby("steps")[column].median().reset_index()
            median_smooth = smooth(agg[column].values, window=5)

            ax.plot(
                agg["steps"], median_smooth,
                label=cond, color=info["color"], linewidth=2,
            )

        # Vertical line at 500 epochs (batch) = 500 steps
        ax.axvline(500, color="black", linestyle="--", alpha=0.3)

        # Cap the y-axis so divergent rules (QPC) don't ruin the scale
        ax.set_ylim(*y_limit)
        
        ax.set_xscale("log")
        ax.set_xlim(1, 200000)
        ax.grid(True, alpha=0.3, which="both")
        
        if col_idx == 0:
            ax.set_ylabel(column.replace('_', ' ').title(), fontsize=12)
        
        if row_idx == 0:
            ax.set_title(LABELS[rule], fontsize=14, fontweight="bold")
        else:
            ax.set_xlabel("Gradient Steps (log scale)", fontsize=10)

if __name__ == "__main__":
    print("Loading histories...")
    df = load_all()

    # 2 rows (Train, Val), 5 cols (Rules)
    fig, axes = plt.subplots(2, 5, figsize=(25, 9), sharex="col")

    print("Plotting Training (Top row)...")
    # Limit train y-axis to see the drop to 0 (QPC will be clipped)
    plot_metric(df, "train_loss", row_idx=0, y_limit=(0, 0.6))

    print("Plotting Validation (Bottom row)...")
    # Limit validation y-axis to see the U-shape before it explodes
    plot_metric(df, "validation_loss", row_idx=1, y_limit=(0, 1.0))

    # Shared legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, 
        loc="lower center", ncol=3, fontsize=14, 
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle(
        "Speed of Acquisition: Training vs Validation Loss by Learning Rule",
        fontsize=18, fontweight="bold", y=0.95
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.88)
    
    out_path = OUT / "grid_train_vs_val.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")