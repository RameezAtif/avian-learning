"""
Extracts the final training and validation loss for each rule under each regime.
Generates a table for the LaTeX document.
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

FILES = {
    "Online": RESULTS / "baseline_online" / "baseline_online_histories.csv",
    "CLS": RESULTS / "baseline_cls" / "baseline_cls_histories.csv",
    "Go-CLS": RESULTS / "rq1_baseline" / "rq1_histories.csv",
}
RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
FRACTION = 0.10 # Mean of the final 10% of epochs

def main():
    rows = []
    for cond, path in FILES.items():
        if not path.exists():
            print(f"Missing: {path}")
            continue
        df = pd.read_csv(path)

        # Standardise column names
        train_col = "train_loss" if "train_loss" in df.columns else "replay_loss"

        # Online uses 'step', CLS/Go-CLS use 'epoch'
        if "epoch" in df.columns:
            time_col = "epoch"
        elif "step" in df.columns:
            time_col = "step"
        else:
            raise ValueError(f"No 'epoch' or 'step' column in {path}")

        for rule in RULES:
            sub = df[df["learning_rule"] == rule]
            if sub.empty:
                continue

            final_trains = []
            final_vals = []

            for seed, group in sub.groupby("seed"):
                group = group.sort_values(time_col)
                n_final = max(1, int(FRACTION * len(group)))
                final_trains.append(group[train_col].tail(n_final).mean())
                final_vals.append(group["validation_loss"].tail(n_final).mean())

            rows.append({
                "Regime": cond,
                "Rule": rule,
                "Train": np.mean(final_trains),
                "Val": np.mean(final_vals),
            })

    df_res = pd.DataFrame(rows)

    # Print text table
    print("=" * 90)
    print(f"{'Rule':<16} | {'Online (Train/Val)':<20} | {'CLS (Train/Val)':<20} | {'Go-CLS (Train/Val)':<20}")
    print("-" * 90)
    for rule in RULES:
        vals = {}
        for cond in ["Online", "CLS", "Go-CLS"]:
            row = df_res[(df_res["Regime"] == cond) & (df_res["Rule"] == rule)]
            if not row.empty:
                vals[cond] = f"{row['Train'].values[0]:.3f} / {row['Val'].values[0]:.3f}"
            else:
                vals[cond] = "n/a"
        print(f"{rule:<16} | {vals['Online']:<20} | {vals['CLS']:<20} | {vals['Go-CLS']:<20}")

    print("=" * 90)

    # Save to CSV for your records
    df_res.to_csv(RESULTS / "baseline_summary_table.csv", index=False)
    print(f"\nSaved: {RESULTS / 'baseline_summary_table.csv'}")

if __name__ == "__main__":
    main()