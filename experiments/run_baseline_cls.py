"""
Baseline B — Complementary Learning Systems, no generalisation gating.

Same as RQ1 but without early stopping based on validation.
Trains for a fixed number of epochs.

Run:
    python experiments/run_baseline_cls.py
"""

from dataclasses import replace
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import ExperimentConfig
from experiments.runner import run_experiment
import pandas as pd


RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
SEEDS = list(range(42, 72))
RESULTS = PROJECT_ROOT / "results" / "baseline_cls"
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    total = len(RULES) * len(SEEDS)
    count = 0

    for rule in RULES:
        for seed in SEEDS:
            count += 1
            print(f"[{count}/{total}] {rule} seed={seed}")

            # patience = max_epochs → never triggers early stop
            # use_snr_gating = False → uniform replay (CLS, not Go-CLS)
            config = replace(
                ExperimentConfig(seed=seed),
                patience=10**6,
                use_snr_gating=False,
            )

            result = run_experiment(
                learning_rule_name=rule,
                config=config,
            )

            rows.append({
                "learning_rule": rule,
                "seed": seed,
                "initial_validation_loss": result.initial_validation_loss,
                "best_validation_loss": result.best_validation_loss,
                "best_epoch": result.best_epoch,
                "epochs_executed": result.epochs_executed,
                "relative_improvement_pct": 100.0 * (
                    1 - result.best_validation_loss / result.initial_validation_loss
                ),
                "stop_reason": result.stop_reason,
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "baseline_cls_summary.csv", index=False)

    print()
    print(df.groupby("learning_rule")["relative_improvement_pct"].agg(
        ["mean", "std", "count"]))
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()