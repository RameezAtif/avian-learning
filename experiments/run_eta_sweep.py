"""
Diagnostic: sweep eta for Pure Hebbian and Anti-Hebbian rules.

This is a control experiment, not a research question. It exists
because the proposal (Risk Factors, Risk 1) commits to a
hyperparameter sweep over eta before recording Hebbian results.

Sweep over a fixed grid of eta values on a single seed, using the
same architecture and data generation as the main RQ1 experiment.

Output: results/diagnostics/eta_sweep.csv

Run from project root:
    python experiments/run_eta_sweep.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np

from experiments.config import ExperimentConfig
from experiments.runner import run_experiment


ETA_GRID = [
    0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5,
]

SEEDS = [42]

RULES_TO_SWEEP = ["hebbian", "anti_hebbian"]


def main():
    output_dir = PROJECT_ROOT / "results" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "eta_sweep.csv"

    rows = []

    for rule in RULES_TO_SWEEP:
        for eta in ETA_GRID:
            

            for seed in SEEDS:

                # Build a config identical to RQ1 but with this eta.
                config = ExperimentConfig(seed=seed)

                # NOTE: run_experiment reads eta from the rule
                # dispatch table, not from config. We therefore
                # monkey-patch the dispatch by importing the rule
                # factory and overriding its behavior for this run.

                # The cleanest approach: use a helper that accepts
                # eta explicitly. See helper below.
                signed_eta = eta if rule == "hebbian" else -eta
                result = run_hebbian_with_eta(
                    rule_name=rule,
                    eta=signed_eta,
                    config=config,
                )

                rows.append({
                    "rule": rule,
                    "eta": eta,
                    "seed": seed,
                    "initial_validation_loss":
                        result.initial_validation_loss,
                    "best_validation_loss":
                        result.best_validation_loss,
                    "best_epoch": result.best_epoch,
                    "epochs_executed": result.epochs_executed,
                    "stop_reason": result.stop_reason,
                    "relative_improvement_percent": (
                        (
                            result.initial_validation_loss
                            - result.best_validation_loss
                        )
                        / result.initial_validation_loss
                        * 100
                    ),
                })

                print(
                    f"{rule:>14}  eta={eta:<6}  "
                    f"best={result.best_validation_loss:.6f}  "
                    f"improvement={rows[-1]['relative_improvement_percent']:.3f}%"
                )

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved: {output_file}")


def run_hebbian_with_eta(rule_name, eta, config):
    """
    Run one experiment with a specific eta value.

    This duplicates a small amount of runner logic so that eta can
    be varied without modifying the main runner. It reuses the same
    data generation, student, notebook, replay, and controller.
    """

    from learning.plasticity import ContinuousPlasticityRule
    from data.teacher import create_teacher, generate_teacher_experiences
    from models.student import Student
    from memory.notebook import SparseHopfieldNotebook
    from consolidation.replay import SleepReplay
    from consolidation.controller import GoCLSController
    from metrics.stability import calculate_windowed_stability
    import numpy as np

    # ---- Data ----
    teacher = create_teacher(config.input_dim, seed=config.seed)
    tutor = generate_teacher_experiences(
        teacher, config.num_tutor_examples,
        1.0 / config.tutor_snr, config.seed + 1, "tutor"
    )
    practice = generate_teacher_experiences(
        teacher, config.num_practice_examples,
        1.0 / config.practice_snr, config.seed + 2, "practice"
    )
    train_x = np.vstack((tutor.x, practice.x))
    train_y = np.concatenate((tutor.y, practice.y))

    evaluation_noise = (
        0.0 if np.isinf(config.evaluation_snr)
        else 1.0 / config.evaluation_snr
    )
    validation = generate_teacher_experiences(
        teacher, config.num_validation_examples,
        evaluation_noise, config.seed + 3, "evaluation"
    )

    # ---- Student ----
    student = Student(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )

    # ---- Notebook ----
    notebook = SparseHopfieldNotebook(
        notebook_dim=config.notebook_dim,
        sparsity=config.notebook_sparsity,
        seed=config.seed,
    )
    notebook.encode_batch(x=train_x, y=train_y)

    # ---- Learning rule with the swept eta ----
    learning_rule = ContinuousPlasticityRule(
        gamma=0.0,
        eta=eta,
        learning_rate=config.learning_rate,
        gradient_clip=config.gradient_clip,
        update_w2=config.update_w2,
    )

    # ---- Replay ----
    replay = SleepReplay(
        student=student,
        notebook=notebook,
        learning_rule=learning_rule,
        replay_cycles=config.replay_cycles,
        replays_per_epoch=config.replays_per_epoch,
        update_w2=config.update_w2,
        replay_mode=config.replay_mode,
    )

    # ---- Controller ----
    controller = GoCLSController(
        replay=replay,
        patience=config.patience,
        min_delta=config.min_delta,
        max_epochs=config.max_epochs,
    )

    initial_validation_loss = controller.validation_loss(
        student, validation.x, validation.y
    )

    consolidation = controller.run(
        x_validation=validation.x,
        y_validation=validation.y,
    )

    # ---- Build result object (minimal fields) ----
    from experiments.runner import ExperimentResult
    import numpy as np

    replay_losses = np.asarray(
        [m.loss for m in consolidation.history], dtype=float
    )
    validation_losses = np.asarray(
        consolidation.validation_losses, dtype=float
    )

    return ExperimentResult(
        learning_rule=rule_name,
        tutor_snr=config.tutor_snr,
        practice_snr=config.practice_snr,
        seed=config.seed,
        replay_losses=replay_losses,
        validation_losses=validation_losses,
        acquisition_epoch=None,
        acquisition_threshold=float("nan"),
        initial_validation_loss=initial_validation_loss,
        stability=None,
        stop_reason=consolidation.stop_reason,
        best_validation_loss=consolidation.best_validation_loss,
        best_epoch=consolidation.best_epoch,
        epochs_executed=consolidation.epochs_executed,
    )


if __name__ == "__main__":
    main()