"""
RQ1 baseline experiment.

Compares the five learning rules under identical experimental conditions:

    1. Gradient Descent
    2. Contrastive Hebbian Learning (current error-feedback surrogate)
    3. Quasi-Predictive Coding
    4. Pure Hebbian
    5. Anti-Hebbian

Each rule is evaluated across multiple random seeds.

Outputs:

    results/rq1_baseline/rq1_summary.csv
    results/rq1_baseline/rq1_histories.csv

The underlying learning algorithms and ExperimentConfig are not modified.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable when this file is run directly:
#
#     python3 experiments/run_rq1.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentResult, run_experiment


# ---------------------------------------------------------------------------
# Experimental protocol
# ---------------------------------------------------------------------------

LEARNING_RULES = [
    "gradient_descent",
    "chl",
    "qpc",
    "hebbian",
    "anti_hebbian",
]

SEEDS = [42, 43, 44, 45, 46]

RESULTS_DIR = PROJECT_ROOT / "results" / "rq1_baseline"

SUMMARY_FILE = RESULTS_DIR / "rq1_summary.csv"
HISTORIES_FILE = RESULTS_DIR / "rq1_histories.csv"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def make_config(seed: int) -> ExperimentConfig:
    """
    Create the RQ1 configuration for one seed.

    All experimental hyperparameters come from the current default
    ExperimentConfig. Only the random seed is changed.
    """
    base_config = ExperimentConfig()

    return replace(
        base_config,
        seed=seed,
    )


def result_to_summary_row(result: ExperimentResult) -> dict:
    """
    Convert an ExperimentResult into one CSV row.
    """

    return {
        "learning_rule": result.learning_rule,
        "tutor_snr": result.tutor_snr,
        "practice_snr": result.practice_snr,
        "seed": result.seed,
        "initial_validation_loss": result.initial_validation_loss,
        "best_validation_loss": result.best_validation_loss,
        "best_epoch": result.best_epoch,
        "acquisition_epoch": result.acquisition_epoch,
        "acquisition_threshold": result.acquisition_threshold,
        "stability": (
            None
            if result.stability is None
            else result.stability.stability_variance
        ),
        "stop_reason": result.stop_reason,
        "epochs_executed": result.epochs_executed,
    }


def result_to_history_rows(result: ExperimentResult) -> list[dict]:
    """
    Convert replay and validation histories into long-format CSV rows.

    One row corresponds to one epoch.
    """

    rows = []

    replay_losses = result.replay_losses
    validation_losses = result.validation_losses

    num_epochs = max(
        len(replay_losses),
        len(validation_losses),
    )

    for epoch_index in range(num_epochs):
        replay_loss = (
            replay_losses[epoch_index]
            if epoch_index < len(replay_losses)
            else None
        )

        validation_loss = (
            validation_losses[epoch_index]
            if epoch_index < len(validation_losses)
            else None
        )

        rows.append(
            {
                "learning_rule": result.learning_rule,
                "seed": result.seed,
                "epoch": epoch_index + 1,
                "replay_loss": replay_loss,
                "validation_loss": validation_loss,
            }
        )

    return rows


def write_summary_csv(results: list[ExperimentResult]) -> None:
    """
    Write one row per experiment.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "learning_rule",
        "tutor_snr",
        "practice_snr",
        "seed",
        "initial_validation_loss",
        "best_validation_loss",
        "best_epoch",
        "acquisition_epoch",
        "acquisition_threshold",
        "stability",
        "stop_reason",
        "epochs_executed",
    ]

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                result_to_summary_row(result)
            )


def write_histories_csv(results: list[ExperimentResult]) -> None:
    """
    Write replay and validation losses for every epoch of every run.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "learning_rule",
        "seed",
        "epoch",
        "replay_loss",
        "validation_loss",
    ]

    with HISTORIES_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            rows = result_to_history_rows(result)

            for row in rows:
                writer.writerow(row)


def print_result(result: ExperimentResult) -> None:
    """
    Print a compact summary after each completed experiment.
    """

    print()
    print("-" * 72)
    print(
        f"Completed: {result.learning_rule} "
        f"(seed={result.seed})"
    )
    print(f"Epochs executed:       {result.epochs_executed}")
    print(f"Initial validation:    {result.initial_validation_loss:.8f}")
    print(f"Best validation:       {result.best_validation_loss:.8f}")
    print(f"Best epoch:            {result.best_epoch}")

    if result.acquisition_epoch is None:
        print("Acquisition epoch:     NOT REACHED")
    else:
        print(
            f"Acquisition epoch:     {result.acquisition_epoch}"
        )

    if result.acquisition_threshold is None:
        print("Acquisition threshold:  None")
    else:
        print(
            f"Acquisition threshold: {result.acquisition_threshold:.8f}"
        )

    if result.stability is None:
        print("Stability:             NOT AVAILABLE")
    else:
        print(
        f"Stability variance:    "
        f"{result.stability.stability_variance:.10f}"
        )

    print(f"Stop reason:           {result.stop_reason}")
    print("-" * 72)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the complete RQ1 experiment.

    Total runs:

        5 learning rules × 5 seeds = 25 runs
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_runs = len(LEARNING_RULES) * len(SEEDS)

    print("=" * 72)
    print("RQ1 BASELINE EXPERIMENT")
    print("=" * 72)
    print()
    print("Learning rules:")
    for rule in LEARNING_RULES:
        print(f"  - {rule}")

    print()
    print(f"Seeds:        {SEEDS}")
    print(f"Total runs:   {total_runs}")
    print()
    print(f"Results directory:")
    print(f"  {RESULTS_DIR}")
    print()
    print("Starting experiments...")
    print()

    results: list[ExperimentResult] = []

    completed_runs = 0
    failed_runs = 0

    overall_start = time.perf_counter()

    for rule in LEARNING_RULES:

        for seed in SEEDS:

            completed_runs += 1

            print("=" * 72)
            print(
                f"RUN {completed_runs}/{total_runs}: "
                f"{rule} | seed={seed}"
            )
            print("=" * 72)

            config = make_config(seed)

            start_time = time.perf_counter()

            try:
                result = run_experiment(
                    config=config,
                    learning_rule_name=rule,
                )

                elapsed = time.perf_counter() - start_time

                results.append(result)

                # Save immediately after a successful experiment.
                write_summary_csv(results)
                write_histories_csv(results)

                print_result(result)

                print(
                    f"Runtime:               {elapsed:.2f} seconds"
                )

                print(
                    f"Saved {len(results)} completed runs."
                )

                print(
                    f"Runtime:               {elapsed:.2f} seconds"
                )

                # results.append(result)

                # # Save after EVERY completed experiment.
                # #
                # # This is intentional. If a later run crashes, we don't
                # # lose all the experiments that already completed.
                # write_summary_csv(results)
                # write_histories_csv(results)

                print()
                print(
                    f"Saved {len(results)} completed runs."
                )

            except Exception as error:
                failed_runs += 1

                elapsed = time.perf_counter() - start_time

                print()
                print("!!! EXPERIMENT FAILED !!!")
                print(
                    f"Rule:    {rule}"
                )
                print(
                    f"Seed:    {seed}"
                )
                print(
                    f"Runtime: {elapsed:.2f} seconds"
                )
                print(
                    f"Error:   {type(error).__name__}: {error}"
                )
                print()

                # Do not silently continue.
                #
                # A failed run needs to be investigated because we don't
                # want an incomplete experimental dataset without noticing.
                raise

    overall_elapsed = time.perf_counter() - overall_start

    # Final save.
    write_summary_csv(results)
    write_histories_csv(results)

    print()
    print("=" * 72)
    print("RQ1 EXPERIMENT COMPLETE")
    print("=" * 72)
    print()
    print(f"Completed runs: {len(results)}")
    print(f"Failed runs:    {failed_runs}")
    print(f"Total runtime:  {overall_elapsed:.2f} seconds")
    print()
    print("Output files:")
    print(f"  {SUMMARY_FILE}")
    print(f"  {HISTORIES_FILE}")
    print()


if __name__ == "__main__":
    main()