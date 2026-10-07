"""
Compare three conditions: Online, CLS, Go-CLS.

Produces:
    Individual figures (one per condition):
        online_improvement.png
        cls_improvement.png
        go_cls_improvement.png

    Joint figures:
        joint_improvement_comparison.png   (grouped bar)
        joint_improvement_heatmap.png      (rules x conditions)
        joint_slope_chart.png              (one line per rule across conditions)

    Tables:
        joint_summary.csv
        joint_statistical_tests.csv

Run:
    python analysis/compare_conditions.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Paths
RESULTS_DIR = PROJECT_ROOT / "results"
ONLINE_FILE = RESULTS_DIR / "baseline_online" / "baseline_online_summary.csv"
CLS_FILE = RESULTS_DIR / "baseline_cls" / "baseline_cls_summary.csv"
GOCLS_FILE = RESULTS_DIR / "rq1_baseline" / "rq1_summary.csv"

OUTPUT_DIR = RESULTS_DIR / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rules
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
CONDITION_COLORS = {
    "Online": "#4477AA",
    "CLS": "#CC6677",
    "Go-CLS": "#228833",
}
CONDITIONS = ["Online", "CLS", "Go-CLS"]


# =====================================================================
# Data loading
# =====================================================================

def _ensure_improvement_column(df):
    if "relative_improvement_pct" in df.columns:
        return df

    if "final_validation_loss" in df.columns:
        df["relative_improvement_pct"] = (
            100.0
            * (1 - df["final_validation_loss"] / df["initial_validation_loss"])
        )
    else:
        df["relative_improvement_pct"] = (
            100.0
            * (1 - df["best_validation_loss"] / df["initial_validation_loss"])
        )
    return df


def load_all():
    frames = []

    online = pd.read_csv(ONLINE_FILE)
    online["condition"] = "Online"
    online = _ensure_improvement_column(online)
    frames.append(online)

    cls = pd.read_csv(CLS_FILE)
    cls["condition"] = "CLS"
    cls = _ensure_improvement_column(cls)
    frames.append(cls)

    gocls = pd.read_csv(GOCLS_FILE)
    gocls["condition"] = "Go-CLS"
    gocls = _ensure_improvement_column(gocls)
    frames.append(gocls)

    return pd.concat(frames, ignore_index=True)


# =====================================================================
# Individual figures — bar chart with error bars
# =====================================================================

def plot_individual(df, condition):
    sub = df[df["condition"] == condition]

    fig, ax = plt.subplots(figsize=(9, 5))

    means, stds, labels, colors = [], [], [], []
    for rule in RULES:
        vals = sub[sub["learning_rule"] == rule][
            "relative_improvement_pct"
        ].dropna().values
        means.append(vals.mean() if len(vals) > 0 else np.nan)
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        labels.append(LABELS[rule])
        colors.append(COLORS[rule])

    x = np.arange(len(RULES))
    ax.bar(
        x, means, yerr=stds, capsize=5,
        color=colors, alpha=0.75, edgecolor="black", linewidth=0.8,
    )

    # Annotate each bar with mean value
    for xi, m in zip(x, means):
        ax.text(
            xi, m + 1.5, f"{m:.1f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative improvement (%)")
    ax.set_title(f"{condition} — mean improvement per rule (n=30 seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8)
    fig.tight_layout()

    fname = f"{condition.lower().replace('-', '_')}_improvement.png"
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# =====================================================================
# Joint figures
# =====================================================================

def plot_grouped_bar(df):
    """Grouped bar chart: 5 rules on x, one bar per condition."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(RULES))
    width = 0.25

    for i, cond in enumerate(CONDITIONS):
        means, stds = [], []
        for rule in RULES:
            vals = df[
                (df["condition"] == cond) & (df["learning_rule"] == rule)
            ]["relative_improvement_pct"].dropna()
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))

        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=cond,
            color=CONDITION_COLORS[cond],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in RULES])
    ax.set_ylabel("Relative improvement (%)")
    ax.set_title("Improvement per rule across conditions (mean ± SD, n=30)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8)
    fig.tight_layout()

    out = OUTPUT_DIR / "joint_improvement_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_heatmap(df):
    """Heatmap: rules x conditions, cell = mean improvement."""
    pivot = df.pivot_table(
        index="learning_rule",
        columns="condition",
        values="relative_improvement_pct",
        aggfunc="mean",
    )
    pivot = pivot.loc[RULES, CONDITIONS]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS)
    ax.set_yticks(np.arange(len(RULES)))
    ax.set_yticklabels([LABELS[r] for r in RULES])

    mid = float(np.nanmean(pivot.values))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                color="white" if val > mid else "black",
                fontsize=11,
            )

    ax.set_title("Mean improvement (%) by rule and condition")
    fig.colorbar(im, ax=ax, label="Improvement (%)")
    fig.tight_layout()

    out = OUTPUT_DIR / "joint_improvement_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_slope_chart(df):
    """
    Slope chart: one line per rule across the three conditions.
    Uses mean improvement per (rule, condition).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    x_positions = np.arange(len(CONDITIONS))

    for rule in RULES:
        means = []
        for cond in CONDITIONS:
            vals = df[
                (df["condition"] == cond) & (df["learning_rule"] == rule)
            ]["relative_improvement_pct"].dropna()
            means.append(vals.mean() if len(vals) > 0 else np.nan)

        ax.plot(
            x_positions, means,
            marker="o", markersize=9,
            linewidth=2.2,
            label=LABELS[rule],
            color=COLORS[rule],
        )

        # Annotate values
        for xi, m in zip(x_positions, means):
            ax.annotate(
                f"{m:.1f}",
                (xi, m),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                color=COLORS[rule],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlim(-0.35, len(CONDITIONS) - 0.65)
    ax.set_ylabel("Mean improvement (%)")
    ax.set_title(
        "Rule ranking flips across regimes\n"
        "(each line = one rule; slope across conditions)"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8)
    fig.tight_layout()

    out = OUTPUT_DIR / "joint_slope_chart.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_ranking_slope(df):
    """
    Ranking slope chart: rank rules within each condition, connect
    rank positions across conditions. Ranks 1 = best (highest improvement).

    This is the clearest way to show that the winner changes across regimes.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    x_positions = np.arange(len(CONDITIONS))

    # Compute rank per rule per condition
    rankings = {rule: [] for rule in RULES}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        means = {}
        for rule in RULES:
            vals = sub[sub["learning_rule"] == rule][
                "relative_improvement_pct"
            ].dropna()
            means[rule] = vals.mean()
        # rank 1 = best
        ordered = sorted(means.items(), key=lambda kv: -kv[1])
        for rank_idx, (rule, _) in enumerate(ordered, start=1):
            rankings[rule].append(rank_idx)

    for rule in RULES:
        ax.plot(
            x_positions, rankings[rule],
            marker="o", markersize=10,
            linewidth=2.4,
            label=LABELS[rule],
            color=COLORS[rule],
        )
        for xi, r in zip(x_positions, rankings[rule]):
            ax.annotate(
                LABELS[rule],
                (xi, r),
                textcoords="offset points",
                xytext=(10, -2),
                fontsize=9,
                color=COLORS[rule],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlim(-0.4, len(CONDITIONS) - 0.3)
    ax.invert_yaxis()   # rank 1 at top
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Rank (1 = best improvement)")
    ax.set_title(
        "Rule ranking by regime\n"
        "(ranks computed within each condition)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = OUTPUT_DIR / "joint_ranking_slope.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =====================================================================
# Tables
# =====================================================================

def summary_table(df):
    rows = []
    for cond in CONDITIONS:
        for rule in RULES:
            vals = df[
                (df["condition"] == cond) & (df["learning_rule"] == rule)
            ]["relative_improvement_pct"].dropna()

            rows.append({
                "condition": cond,
                "learning_rule": rule,
                "learning_rule_label": LABELS[rule],
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(ddof=1),
                "median": vals.median(),
                "min": vals.min(),
                "max": vals.max(),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_DIR / "joint_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'joint_summary.csv'}")
    return out_df


def statistical_tests(df):
    comparisons = [
        ("Online", "CLS"),
        ("Online", "Go-CLS"),
        ("CLS", "Go-CLS"),
    ]
    rows = []

    for rule in RULES:
        for cond_a, cond_b in comparisons:
            a = df[(df["condition"] == cond_a) & (df["learning_rule"] == rule)]
            b = df[(df["condition"] == cond_b) & (df["learning_rule"] == rule)]

            merged = pd.merge(
                a[["seed", "relative_improvement_pct"]],
                b[["seed", "relative_improvement_pct"]],
                on="seed",
                suffixes=("_a", "_b"),
            )

            if len(merged) < 2:
                continue

            diff = (
                merged["relative_improvement_pct_a"]
                - merged["relative_improvement_pct_b"]
            )

            try:
                stat, p = stats.wilcoxon(diff)
            except ValueError:
                stat, p = np.nan, np.nan

            rows.append({
                "learning_rule": rule,
                "learning_rule_label": LABELS[rule],
                "condition_a": cond_a,
                "condition_b": cond_b,
                "mean_diff_a_minus_b": diff.mean(),
                "n_pairs": len(diff),
                "wilcoxon_p": p,
            })

    out_df = pd.DataFrame(rows)
    out_df["bonferroni_p"] = np.minimum(
        out_df["wilcoxon_p"] * len(out_df), 1.0
    )
    out_df.to_csv(OUTPUT_DIR / "joint_statistical_tests.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'joint_statistical_tests.csv'}")
    return out_df


# =====================================================================
# Main
# =====================================================================

def main():
    print("Loading conditions...")
    df = load_all()
    print(f"  total runs: {len(df)}")
    print(f"  conditions: {sorted(df['condition'].unique())}")
    print()

    print("Individual figures...")
    for cond in CONDITIONS:
        plot_individual(df, cond)
    print()

    print("Joint figures...")
    plot_grouped_bar(df)
    plot_heatmap(df)
    plot_slope_chart(df)
    plot_ranking_slope(df)
    print()

    print("Tables...")
    summary = summary_table(df)
    tests = statistical_tests(df)
    print()

    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(summary.to_string(index=False))
    print()

    print("=" * 72)
    print("STATISTICAL TESTS")
    print("=" * 72)
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()