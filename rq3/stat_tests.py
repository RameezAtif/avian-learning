"""
Pairwise Wilcoxon signed-rank tests across rules for RQ3.

Run:
    python rq3/stat_tests.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "rq3" / "results"

RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
LABELS = {
    "gradient_descent": "GD",
    "chl": "CHL",
    "qpc": "QPC",
    "hebbian": "Hebbian",
    "anti_hebbian": "Anti-Hebbian",
}


def main():
    summary = pd.read_csv(RESULTS / "rq3_summary.csv")

    # Pivot: rows = seed, columns = rule, values = improvement
    pivot = summary.pivot(
        index="seed",
        columns="learning_rule",
        values="improvement_pct",
    )

    print("=" * 72)
    print("RQ3 — WILCOXON SIGNED-RANK TESTS (paired by seed, n=10)")
    print("=" * 72)
    print()

    n_pairs = len(RULES) * (len(RULES) - 1) // 2
    rows = []

    for i, rule_a in enumerate(RULES):
        for rule_b in RULES[i + 1:]:
            diff = pivot[rule_a] - pivot[rule_b]

            try:
                stat, p = stats.wilcoxon(diff)
            except ValueError:
                stat, p = float("nan"), float("nan")

            bonferroni_p = min(p * n_pairs, 1.0)

            rows.append({
                "rule_a": rule_a,
                "rule_b": rule_b,
                "mean_diff": diff.mean(),
                "std_diff": diff.std(ddof=1),
                "wilcoxon_stat": stat,
                "p_value": p,
                "bonferroni_p": bonferroni_p,
                "significant_bonferroni": bonferroni_p < 0.05,
            })

    df = pd.DataFrame(rows)

    # Print
    for _, r in df.iterrows():
        marker = "  ← significant" if r["significant_bonferroni"] else ""
        print(
            f"{LABELS[r['rule_a']]:>12} vs {LABELS[r['rule_b']]:<12}  "
            f"mean_diff={r['mean_diff']:>+7.2f}  "
            f"p={r['p_value']:.4f}  "
            f"Bonf p={r['bonferroni_p']:.4f}{marker}"
        )

    print()
    df.to_csv(RESULTS / "rq3_statistical_tests.csv", index=False)
    print(f"Saved: {RESULTS / 'rq3_statistical_tests.csv'}")


if __name__ == "__main__":
    main()