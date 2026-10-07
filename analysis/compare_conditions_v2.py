"""
Comparison figures designed around the three specific questions:

Q1: CLS vs Go-CLS       — does gating help?
Q2: Online vs CLS       — does batch replay help over streaming?
Q3: Online vs Go-CLS    — same question, full pipeline

Produces:
    schematic.png                       — pipeline diagram for the 3 conditions
    paired_scatter_q1.png               — CLS vs Go-CLS, one panel per rule
    paired_scatter_q2.png               — Online vs CLS
    paired_scatter_q3.png               — Online vs Go-CLS
    slope_plot_q1.png                   — one seed per line, CLS -> Go-CLS
    slope_plot_q2.png                   — Online -> CLS
    slope_plot_q3.png                   — Online -> Go-CLS
    forest_q1_q2_q3.png                 — summary of all three questions

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import stats


RESULTS_DIR = PROJECT_ROOT / "results"
ONLINE_FILE = RESULTS_DIR / "baseline_online" / "baseline_online_summary.csv"
CLS_FILE = RESULTS_DIR / "baseline_cls" / "baseline_cls_summary.csv"
GOCLS_FILE = RESULTS_DIR / "rq1_baseline" / "rq1_summary.csv"
OUTPUT_DIR = RESULTS_DIR / "comparison_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_paired():
    """Load all three conditions, merge on (learning_rule, seed)."""
    online = _ensure_improvement(pd.read_csv(ONLINE_FILE))
    online = online[["learning_rule", "seed", "relative_improvement_pct"]]
    online = online.rename(columns={"relative_improvement_pct": "online"})

    cls = _ensure_improvement(pd.read_csv(CLS_FILE))
    cls = cls[["learning_rule", "seed", "relative_improvement_pct"]]
    cls = cls.rename(columns={"relative_improvement_pct": "cls"})

    gocls = _ensure_improvement(pd.read_csv(GOCLS_FILE))
    gocls = gocls[["learning_rule", "seed", "relative_improvement_pct"]]
    gocls = gocls.rename(columns={"relative_improvement_pct": "gocls"})

    merged = online.merge(cls, on=["learning_rule", "seed"])
    merged = merged.merge(gocls, on=["learning_rule", "seed"])
    return merged


# =====================================================================
# Schematic — pipeline diagram for 3 conditions
# =====================================================================

def plot_schematic():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    configs = [
        ("Online", "#4477AA", [
            "Stream samples",
            "1 sample  →  update",
            "Discard sample",
            "Repeat",
        ]),
        ("CLS", "#CC6677", [
            "Store batch in notebook",
            "Replay N samples",
            "1 batch  →  update",
            "Repeat (no gating)",
        ]),
        ("Go-CLS", "#228833", [
            "Store batch in notebook",
            "Replay N samples",
            "1 batch  →  update",
            "Stop when validation plateaus",
        ]),
    ]

    for ax, (name, color, steps) in zip(axes, configs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(name, fontsize=14, fontweight="bold", color=color)

        y = 8.5
        for step in steps:
            box = mpatches.FancyBboxPatch(
                (0.5, y - 0.6), 9, 1.2,
                boxstyle="round,pad=0.08",
                facecolor=color, alpha=0.2, edgecolor=color, linewidth=2,
            )
            ax.add_patch(box)
            ax.text(5, y, step, ha="center", va="center", fontsize=10)
            y -= 1.9

    fig.suptitle(
        "Three training conditions — pipeline overview",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "schematic.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =====================================================================
# Paired scatter — one panel per rule, y=x diagonal
# =====================================================================

def plot_paired_scatter(merged, cond_a, cond_b, title, fname):
    """
    Rows = 5 rules. Each panel:
        x-axis = condition A improvement
        y-axis = condition B improvement
        one dot per seed
        diagonal y=x
        annotate mean shift
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=False)

    for ax, rule in zip(axes, RULES):
        sub = merged[merged["learning_rule"] == rule]
        x = sub[cond_a].values
        y = sub[cond_b].values

        # Common range
        lo = min(x.min(), y.min(), 0) - 5
        hi = max(x.max(), y.max()) + 5

        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)

        ax.scatter(x, y, s=60, alpha=0.7, color=COLORS[rule],
                   edgecolor="black", linewidth=0.6)

        # Mean shift arrow
        xm, ym = x.mean(), y.mean()
        ax.plot(xm, ym, marker="D", color="black", markersize=8, zorder=5)
        ax.annotate(
            f"Δ={ym - xm:+.1f}",
            (xm, ym),
            textcoords="offset points",
            xytext=(10, -10),
            fontsize=10, fontweight="bold",
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{cond_a} improvement (%)")
        ax.set_ylabel(f"{cond_b} improvement (%)")
        ax.set_title(LABELS[rule])
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =====================================================================
# Slope plot — one line per seed
# =====================================================================

def plot_slope(merged, cond_a, cond_b, title, fname):
    """
    Rows = 5 rules. Each panel:
        x-axis = [cond_a, cond_b]
        one line per seed connecting its two values
        bold black line = mean across seeds
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

    x_positions = [0, 1]

    for ax, rule in zip(axes, RULES):
        sub = merged[merged["learning_rule"] == rule]

        for _, row in sub.iterrows():
            ax.plot(
                x_positions,
                [row[cond_a], row[cond_b]],
                color=COLORS[rule], alpha=0.3, linewidth=1,
            )

        # Mean line
        mean_a = sub[cond_a].mean()
        mean_b = sub[cond_b].mean()
        ax.plot(
            x_positions, [mean_a, mean_b],
            color="black", linewidth=3, marker="D",
            markersize=10, zorder=10,
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([cond_a, cond_b])
        ax.set_title(LABELS[rule])
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.8)

    axes[0].set_ylabel("Relative improvement (%)")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =====================================================================
# Forest plot — summary of all three questions
# =====================================================================

def plot_forest(merged):
    """
    For each rule and each condition pair, compute mean difference and
    bootstrap 95% CI. Plot as a forest plot: rules on y-axis, three
    comparisons as coloured markers.
    """
    comparisons = [
        ("cls", "gocls", "CLS → Go-CLS", "tab:orange"),
        ("online", "cls", "Online → CLS", "tab:blue"),
        ("online", "gocls", "Online → Go-CLS", "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    n_rules = len(RULES)
    y_ticks = np.arange(n_rules)
    offset = 0.22

    for i, (a, b, label, color) in enumerate(comparisons):
        means, los, his = [], [], []
        for rule in RULES:
            sub = merged[merged["learning_rule"] == rule]
            diffs = sub[b].values - sub[a].values
            mean = diffs.mean()

            # Bootstrap CI
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
        ax.errorbar(
            means, y_positions,
            xerr=[np.array(means) - np.array(los),
                  np.array(his) - np.array(means)],
            fmt="o", color=color, label=label,
            capsize=4, markersize=8,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([LABELS[r] for r in RULES])
    ax.set_xlabel("Mean improvement difference (pp)")
    ax.set_title("Mean difference per comparison, with 95% bootstrap CI")
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

def summary_by_question(merged):
    """
    For each question and each rule, report:
      mean difference, std, Wilcoxon p, Bonferroni p, direction
    """
    comparisons = [
        ("cls", "gocls", "Q1: CLS vs Go-CLS"),
        ("online", "cls", "Q2: Online vs CLS"),
        ("online", "gocls", "Q3: Online vs Go-CLS"),
    ]

    rows = []
    for a, b, label in comparisons:
        for rule in RULES:
            sub = merged[merged["learning_rule"] == rule]
            diffs = sub[b].values - sub[a].values

            try:
                _, p = stats.wilcoxon(diffs)
            except ValueError:
                p = np.nan

            rows.append({
                "question": label,
                "learning_rule": rule,
                "learning_rule_label": LABELS[rule],
                "n_pairs": len(diffs),
                "mean_diff": diffs.mean(),
                "std_diff": diffs.std(ddof=1),
                "wilcoxon_p": p,
                "direction": (
                    "B > A" if diffs.mean() > 0 else "B < A"
                ),
            })

    out = pd.DataFrame(rows)
    out["bonferroni_p"] = np.minimum(out["wilcoxon_p"] * len(out), 1.0)
    out.to_csv(OUTPUT_DIR / "questions_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'questions_summary.csv'}")
    return out


# =====================================================================
# Main
# =====================================================================

def main():
    print("Loading paired data...")
    merged = load_paired()
    print(f"  merged rows: {len(merged)}")
    print()

    plot_schematic()
    print()

    plot_paired_scatter(
        merged, "cls", "gocls",
        "Q1 — Does Go-CLS gating help over CLS? "
        "(points near the diagonal = no difference)",
        "paired_scatter_q1.png",
    )
    plot_paired_scatter(
        merged, "online", "cls",
        "Q2 — Does batch replay help over streaming? "
        "(Online vs CLS)",
        "paired_scatter_q2.png",
    )
    plot_paired_scatter(
        merged, "online", "gocls",
        "Q3 — Does batch replay help over streaming? "
        "(Online vs Go-CLS)",
        "paired_scatter_q3.png",
    )
    print()

    plot_slope(
        merged, "cls", "gocls",
        "Q1 — CLS → Go-CLS. "
        "If lines are flat, gating does nothing.",
        "slope_plot_q1.png",
    )
    plot_slope(
        merged, "online", "cls",
        "Q2 — Online → CLS. "
        "If lines slope up, batch replay helps.",
        "slope_plot_q2.png",
    )
    plot_slope(
        merged, "online", "gocls",
        "Q3 — Online → Go-CLS.",
        "slope_plot_q3.png",
    )
    print()

    plot_forest(merged)
    print()

    summary = summary_by_question(merged)
    print()
    print("=" * 90)
    print("SUMMARY OF THE THREE QUESTIONS")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()