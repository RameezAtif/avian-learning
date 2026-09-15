"""
Diagnostic: measure ||W1_final - W1_initial|| and
||W2_final - W2_initial|| for each learning rule.

This confirms whether the W1 Hebbian term is contributing
meaningfully to training, or whether W2 supervised learning
dominates the outcome.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np

from experiments.config import ExperimentConfig
from data.teacher import create_teacher, generate_teacher_experiences
from models.student import Student
from memory.notebook import SparseHopfieldNotebook
from consolidation.replay import SleepReplay
from consolidation.controller import GoCLSController
from learning.plasticity import ContinuousPlasticityRule
from learning.gradient_descent import GradientDescentRule
from learning.chl import ContrastiveHebbianRule


SEED = 42

RULES = [
    "gradient_descent",
    "chl",
    "qpc",
    "hebbian",
    "anti_hebbian",
]


def create_rule(rule_name, config):
    if rule_name == "gradient_descent":
        return GradientDescentRule(
            learning_rate=config.learning_rate,
            update_w2=config.update_w2,
            gradient_clip=config.gradient_clip,
        )
    if rule_name == "chl":
        return ContrastiveHebbianRule(
            learning_rate=config.learning_rate,
            feedback_strength=1.0,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "qpc":
        return ContinuousPlasticityRule(
            gamma=-1.0, eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "hebbian":
        return ContinuousPlasticityRule(
            gamma=0.0, eta=0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "anti_hebbian":
        return ContinuousPlasticityRule(
            gamma=0.0, eta=-0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    raise ValueError(rule_name)


def main():
    config = ExperimentConfig(seed=SEED)

    output_dir = PROJECT_ROOT / "results" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "weight_movement.csv"

    rows = []

    for rule_name in RULES:

        # ---- Data (same as runner) ----
        teacher = create_teacher(config.input_dim, seed=config.seed)
        tutor = generate_teacher_experiences(
            teacher, config.num_tutor_examples,
            1.0 / config.tutor_snr, config.seed + 1, "tutor",
        )
        practice = generate_teacher_experiences(
            teacher, config.num_practice_examples,
            1.0 / config.practice_snr, config.seed + 2, "practice",
        )
        train_x = np.vstack((tutor.x, practice.x))
        train_y = np.concatenate((tutor.y, practice.y))

        evaluation_noise = (
            0.0 if np.isinf(config.evaluation_snr)
            else 1.0 / config.evaluation_snr
        )
        validation = generate_teacher_experiences(
            teacher, config.num_validation_examples,
            evaluation_noise, config.seed + 3, "evaluation",
        )

        # ---- Student ----
        student = Student(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
        )

        # ---- Snapshot initial weights ----
        w1_init = student.W1.copy()
        w2_init = student.W2.copy()

        # ---- Notebook ----
        notebook = SparseHopfieldNotebook(
            notebook_dim=config.notebook_dim,
            sparsity=config.notebook_sparsity,
            seed=config.seed,
        )
        notebook.encode_batch(x=train_x, y=train_y)

        # ---- Rule ----
        rule = create_rule(rule_name, config)

        # ---- Replay ----
        replay = SleepReplay(
            student=student,
            notebook=notebook,
            learning_rule=rule,
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
        initial_loss = controller.validation_loss(
            student, validation.x, validation.y,
        )
        consolidation = controller.run(
            x_validation=validation.x,
            y_validation=validation.y,
        )

        # ---- Measure movement ----
        w1_final = student.W1
        w2_final = student.W2

        w1_change_norm = float(np.linalg.norm(w1_final - w1_init))
        w2_change_norm = float(np.linalg.norm(w2_final - w2_init))

        w1_init_norm = float(np.linalg.norm(w1_init))
        w2_init_norm = float(np.linalg.norm(w2_init))

        relative_w1 = w1_change_norm / (w1_init_norm + 1e-12)
        relative_w2 = w2_change_norm / (w2_init_norm + 1e-12)

        rows.append({
            "rule": rule_name,
            "w1_change_norm": w1_change_norm,
            "w2_change_norm": w2_change_norm,
            "w1_relative_change": relative_w1,
            "w2_relative_change": relative_w2,
            "initial_validation_loss": initial_loss,
            "best_validation_loss": consolidation.best_validation_loss,
            "relative_improvement_percent": (
                (initial_loss - consolidation.best_validation_loss)
                / initial_loss * 100
            ),
            "epochs_executed": consolidation.epochs_executed,
        })

        print(
            f"{rule_name:>16}  "
            f"|ΔW1|={w1_change_norm:>10.4f}  "
            f"|ΔW2|={w2_change_norm:>10.4f}  "
            f"ratio={w1_change_norm / (w2_change_norm + 1e-12):>10.6f}  "
            f"improvement={rows[-1]['relative_improvement_percent']:>7.3f}%"
        )

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()