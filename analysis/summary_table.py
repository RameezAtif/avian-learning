"""
Summary table: final training and validation loss per rule per condition.

Produces:
    results/baseline_plots/summary_table.csv
    Console-printed LaTeX table

Run:
    python analysis/summary_table.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


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

# Histories for each condition
CONDITIONS = {
    "Online":  RESULTS / "baseline_online" / "baseline_online_histories.csv",
    "CLS":     RESULTS / "baseline_cls" / "baseline_cls_histories.csv",
    "Go-CLS":  RESULTS / "rq1_baseline" / "rq1_histories.csv",
}

# Mean over the last 10% of epochs = "final performance"
FINAL_FRACTION = 0.10


def compute_final_losses():
    """
    For each (condition, rule), compute:
        - mean final training loss across seeds
        - mean final validation loss across seeds
    """
    rows = []

    for cond, path in CONDITIONS.items():
        if not path.exists():
            print(f"[skip] {path} not found")
            continue

        df = pd.read_csv(path)

        # Column name for training loss can be "train_loss" (online)
        # or "replay_loss" (CLS, Go-CLS). Detect automatically.
        if "train_loss" in df.columns:
            train_col = "train_loss"
        elif "replay_loss" in df.columns:
            train_col = "replay_loss"
        else:
            print(f"[skip] {path}: no training loss column")
            continue

        for rule in RULES:
            sub = df[df["learning_rule"] == rule]
            if sub.empty:
                continue

            # Last 10% of epochs per seed
            per_seed_train = []
            per_seed_val = []

            for seed, group in sub.groupby("seed"):
                group = group.sort_values("epoch")
                n_final = max(1, int(FINAL_FRACTION * len(group)))
                train_tail = group[train_col].tail(n_final).mean()
                val_tail = group["validation_loss"].tail(n_final).mean()
                per_seed_train.append(train_tail)
                per_seed_val.append(val_tail)

            rows.append({
                "condition": cond,
                "learning_rule": rule,
                "learning_rule_label": LABELS[rule],
                "n_seeds": len(per_seed_train),
                "train_loss_mean": float(np.mean(per_seed_train)),
                "train_loss_std": float(np.std(per_seed_train, ddof=1)),
                "val_loss_mean": float(np.mean(per_seed_val)),
                "val_loss_std": float(np.std(per_seed_val, ddof=1)),
            })

    return pd.DataFrame(rows)


def format_latex_table(df):
    """
    Produce the LaTeX table with colour-coded entries.
    Format: | Rule | Training Loss | Validation Loss |
    Each cell: Online / CLS / Go-CLS (colour-coded).
    """
    # Pivot to (rule, metric) with condition columns
    pivot = df.pivot_table(
        index="learning_rule",
        columns="condition",
        values=["train_loss_mean", "val_loss_mean"],
    )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Final training and validation loss per rule "
                 r"per training regime. Each cell shows "
                 r"\textcolor{online}{Online} / "
                 r"\textcolor{cls}{CLS} / "
                 r"\textcolor{gocls}{Go-CLS}. Values are the mean over "
                 r"the last 10\% of training epochs, averaged across "
                 r"30 seeds.}")
    lines.append(r"\label{tab:summary_losses}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Rule} & \textbf{Training Loss} & \textbf{Validation Loss} \\")
    lines.append(r"\midrule")

    for rule in RULES:
        if rule not in pivot.index:
            continue
        train = pivot.loc[rule, "train_loss_mean"]
        val = pivot.loc[rule, "val_loss_mean"]

        tr_online = train.get("Online", np.nan)
        tr_cls = train.get("CLS", np.nan)
        tr_gocls = train.get("Go-CLS", np.nan)
        val_online = val.get("Online", np.nan)
        val_cls = val.get("CLS", np.nan)
        val_gocls = val.get("Go-CLS", np.nan)

        rule_label = LABELS[rule]
        # Bold the lowest validation loss per row
        val_candidates = {
            "Online": val_online,
            "CLS": val_cls,
            "Go-CLS": val_gocls,
        }
        best_val = min(
            (v for v in val_candidates.values() if not np.isnan(v)),
            default=np.nan,
        )

        def fmt(x):
            return f"{x:.3f}" if not np.isnan(x) else "--"

        def bold_if_best(x):
            if np.isnan(x) or np.isnan(best_val):
                return fmt(x)
            return r"\textbf{" + fmt(x) + r"}" if abs(x - best_val) < 1e-9 else fmt(x)

        train_cell = (
            rf"\textcolor{{online}}{{{fmt(tr_online)}}} / "
            rf"\textcolor{{cls}}{{{fmt(tr_cls)}}} / "
            rf"\textcolor{{gocls}}{{{fmt(tr_gocls)}}}"
        )
        val_cell = (
            rf"\textcolor{{online}}{{{bold_if_best(val_online)}}} / "
            rf"\textcolor{{cls}}{{{bold_if_best(val_cls)}}} / "
            rf"\textcolor{{gocls}}{{{bold_if_best(val_gocls)}}}"
        )

        lines.append(f"{rule_label} & {train_cell} & {val_cell} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    df = compute_final_losses()
    if df.empty:
        print("No data found.")
        return

    df.to_csv(OUT / "summary_table.csv", index=False)
    print(f"Saved CSV: {OUT / 'summary_table.csv'}")
    print()
    print("=" * 72)
    print("SUMMARY — final training and validation loss")
    print("=" * 72)
    print(df.to_string(index=False))
    print()

    latex = format_latex_table(df)
    print("=" * 72)
    print("LaTeX TABLE")
    print("=" * 72)
    print(latex)
    print()

    with open(OUT / "summary_table.tex", "w") as f:
        f.write(latex)
    print(f"Saved LaTeX: {OUT / 'summary_table.tex'}")


if __name__ == "__main__":
    main()