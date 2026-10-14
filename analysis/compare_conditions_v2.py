"""
Compare four conditions: Online (matched), Online (natural), CLS, Go-CLS.

Produces:
    Individual figures (one per condition):
        online_matched_improvement.png
        online_natural_improvement.png
        cls_improvement.png
        go_cls_improvement.png

    Joint figures:
        joint_improvement_comparison.png   (grouped bar, 4 conditions)
        joint_improvement_heatmap.png      (rules x conditions)
        joint_slope_chart.png              (one line per rule)
        joint_ranking_slope.png            (rank flip)
        forest_q1_q2_q3.png                (summary with bootstrap CI)

    Tables:
        joint_summary.csv
        joint_statistical_tests.csv

Run:
    python analysis/compare_conditions_v2.py
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

# The four conditions
ONLINE_MATCHED_FILE = RESULTS_DIR / "baseline_online_matched" / "baseline_online_matched_summary.csv"
ONLINE_NATURAL_FILE = RESULTS_DIR / "baseline_online" / "baseline_online_summary.csv"
CLS_FILE            = RESULTS_DIR / "baseline_cls" / "baseline_cls_summary.csv"
GOCLS_FILE          = RESULTS_DIR / "rq1_baseline" / "rq1_summary.csv"

OUTPUT_DIR = RESULTS_DIR / "comparison_v2"
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

# Four conditions with distinct colors
CONDITIONS = ["Online (matched)", "Online (natural)", "CLS", "Go-CLS"]
CONDITION_COLORS = {
    "Online (matched)": "#88CCEE",
    "Online (natural)": "#4477AA",
    "CLS": "#CC6677",
    "Go-CLS": "#228833",
}


# =====================================================================
# Load
# =====================================================================

def _ensure_improvement(df):
    if "relative_improvement_pct" in df.columns:
        return df
    if "final_validation_loss" in df.columns:
        df["relative_improvement_pct"] = (
            100.0 * (1 - df["final_validation_loss"] / df["initial_validation_loss"])
        )
    else:
        df["relative_improvement_pct"] = (
            100.0 * (1 - df["best_validation_loss"] / df["initial_validation_loss"])
        )
    return df


def load_all():
    frames = []

    om = _ensure_improvement(pd.read_csv(ONLINE_MATCHED_FILE))
    om["condition"] = "Online (matched)"
    frames.append(om)

    on = _ensure_improvement(pd.read_csv(ONLINE_NATURAL_FILE))
    on["condition"] = "Online (natural)"
    frames.append(on)

    cls = _ensure_improvement(pd.read_csv(CLS_FILE))
    cls["condition"] = "CLS"
    frames.append(cls)

    gocls = _ensure_improvement(pd.read_csv(GOCLS_FILE))
    gocls["condition"] = "Go-CLS"
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

    fname = f"{condition.lower().replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')}_improvement.png"
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# =====================================================================
# Joint figures
# =====================================================================

def plot_grouped_bar(df):
    """Grouped bar chart: 5 rules on x, 4 bars per rule."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    x = np.arange(len(RULES))
    n_conds = len(CONDITIONS)
    total_width = 0.8
    width = total_width / n_conds

    for i, cond in enumerate(CONDITIONS):
        means, stds = [], []
        for rule in RULES:
            vals = df[
                (df["condition"] == cond) & (df["learning_rule"] == rule)
            ]["relative_improvement_pct"].dropna()
            means.append(vals.mean())
      

        offset = (i - (n_conds - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            label=cond,
            color=CONDITION_COLORS[cond],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in RULES])
    ax.set_ylabel("Relative improvement (%)")
    ax.set_title(
        "Improvement per rule across four conditions (mean ± SD, n=30)"
    )
    ax.legend(loc="upper right")
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

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, rotation=15, ha="right")
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
                fontsize=10,
            )

    ax.set_title("Mean improvement (%) by rule and condition")
    fig.colorbar(im, ax=ax, label="Improvement (%)")
    fig.tight_layout()

    out = OUTPUT_DIR / "joint_improvement_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_slope_chart(df):
    """Slope chart: one line per rule across four conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

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

        for xi, m in zip(x_positions, means):
            ax.annotate(
                f"{m:.1f}",
                (xi, m),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=8,
                color=COLORS[rule],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CONDITIONS, rotation=15, ha="right")
    ax.set_xlim(-0.35, len(CONDITIONS) - 0.65)
    ax.set_ylabel("Mean improvement (%)")
    ax.set_title(
        "Each rule's improvement across the four conditions"
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
    """Ranking per condition; lines connect rule positions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(len(CONDITIONS))
    rankings = {rule: [] for rule in RULES}

    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        means = {}
        for rule in RULES:
            vals = sub[sub["learning_rule"] == rule][
                "relative_improvement_pct"
            ].dropna()
            means[rule] = vals.mean()
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
    ax.set_xticklabels(CONDITIONS, rotation=15, ha="right")
    ax.set_xlim(-0.4, len(CONDITIONS) - 0.3)
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Rank (1 = best improvement)")
    ax.set_title("Rule ranking by condition")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = OUTPUT_DIR / "joint_ranking_slope.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_forest(df):
    """
    Forest plot: for each rule, show mean differences for key comparisons.
    """
    comparisons = [
        ("Online (matched)", "CLS", "Matched online vs CLS", "tab:blue"),
        ("Online (matched)", "Go-CLS", "Matched online vs Go-CLS", "tab:green"),
        ("CLS", "Go-CLS", "CLS vs Go-CLS", "tab:orange"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_ticks = np.arange(len(RULES))
    offset = 0.25

    for i, (a, b, label, color) in enumerate(comparisons):
        means, los, his = [], [], []
        for rule in RULES:
            sub = df[df["learning_rule"] == rule]
            va = sub[sub["condition"] == a].set_index("seed")[
                "relative_improvement_pct"
            ]
            vb = sub[sub["condition"] == b].set_index("seed")[
                "relative_improvement_pct"
            ]
            common = va.index.intersection(vb.index)
            if len(common) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue

            diffs = (vb.loc[common] - va.loc[common]).values
            mean = diffs.mean()

            rng = np.random.default_rng(42)
            boots = [
                rng.choice(diffs, size=len(diffs), replace=True).mean()
                for _ in range(2000)
            ]
            lo, hi = np.percentile(boots, [2.5, 97.5])

            means.append(mean)
            los.append(lo)
            his.append(hi)

        y_positions = y_ticks + (i - 1) * offset
        valid = ~np.isnan(means)
        ax.errorbar(
            np.array(means)[valid], y_positions[valid],
            xerr=[
                (np.array(means) - np.array(los))[valid],
                (np.array(his) - np.array(means))[valid],
            ],
            fmt="o", color=color, label=label,
            capsize=4, markersize=8,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([LABELS[r] for r in RULES])
    ax.set_xlabel("Mean improvement difference (B − A, pp)")
    ax.set_title(
        "Mean difference per comparison, with 95% bootstrap CI"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()

    out = OUTPUT_DIR / "forest_q1_q2_q3.png"
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
    """Paired Wilcoxon for each rule across key comparisons."""
    comparisons = [
        ("Online (matched)", "CLS"),
        ("Online (matched)", "Go-CLS"),
        ("CLS", "Go-CLS"),
        ("Online (matched)", "Online (natural)"),
    ]
    rows = []

    for rule in RULES:
        for a, b in comparisons:
            sub = df[df["learning_rule"] == rule]
            va = sub[sub["condition"] == a].set_index("seed")[
                "relative_improvement_pct"
            ]
            vb = sub[sub["condition"] == b].set_index("seed")[
                "relative_improvement_pct"
            ]
            common = va.index.intersection(vb.index)
            if len(common) < 2:
                continue

            diffs = (vb.loc[common] - va.loc[common]).values

            try:
                stat, p = stats.wilcoxon(diffs)
            except ValueError:
                p = np.nan

            rows.append({
                "learning_rule": rule,
                "learning_rule_label": LABELS[rule],
                "condition_a": a,
                "condition_b": b,
                "mean_diff_b_minus_a": diffs.mean(),
                "n_pairs": len(common),
                "wilcoxon_p": p,
            })

    out = pd.DataFrame(rows)
    out["bonferroni_p"] = np.minimum(out["wilcoxon_p"] * len(out), 1.0)
    out.to_csv(OUTPUT_DIR / "joint_statistical_tests.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'joint_statistical_tests.csv'}")
    return out


# =====================================================================
# Main
# =====================================================================

def main():
    print("Loading all conditions...")
    df = load_all()
    print(f"  total runs: {len(df)}")
    print(f"  conditions: {sorted(df['condition'].unique())}")
    print()

    # Individual
    for cond in CONDITIONS:
        plot_individual(df, cond)
    print()

    # Joint
    plot_grouped_bar(df)
    plot_heatmap(df)
    plot_slope_chart(df)
    plot_ranking_slope(df)
    plot_forest(df)
    print()

    # Tables
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