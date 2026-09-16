import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from consolidation.controller import GoCLSController
from consolidation.replay import SleepReplay
from data.teacher import create_teacher, generate_teacher_experiences
from experiments.config import ExperimentConfig
from learning.chl import ContrastiveHebbianRule
from learning.gradient_descent import GradientDescentRule
from learning.plasticity import ContinuousPlasticityRule
from memory.notebook import SparseHopfieldNotebook
from metrics.stability import calculate_windowed_stability
from models.deep_student import DeepStudent

RESULTS_DIR = PROJECT_ROOT / "results" / "rq2_depth"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
SEEDS = [42, 43, 44, 45, 46]


def create_learning_rule(learning_rule_name: str, config: ExperimentConfig):
    if learning_rule_name == "gradient_descent":
        return GradientDescentRule(
            learning_rate=config.learning_rate,
            update_w2=config.update_w2,
            gradient_clip=config.gradient_clip,
        )

    if learning_rule_name == "chl":
        return ContrastiveHebbianRule(
            learning_rate=config.learning_rate,
            feedback_strength=1.0,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )

    if learning_rule_name == "qpc":
        return ContinuousPlasticityRule(
            gamma=-1.0,
            eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )

    if learning_rule_name == "hebbian":
        return ContinuousPlasticityRule(
            gamma=0.0,
            eta=0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )

    if learning_rule_name == "anti_hebbian":
        return ContinuousPlasticityRule(
            gamma=0.0,
            eta=-0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )

    raise ValueError(f"Unknown learning rule: {learning_rule_name}")


def run_rq2():
    summary_records = []
    history_records = []

    print("=" * 72)
    print("STARTING RQ2: STUDENT DEPTH EXPERIMENT")
    print("=" * 72)

    run_idx = 1
    total_runs = len(RULES) * len(SEEDS)

    for rule_name in RULES:
        for seed in SEEDS:
            config = ExperimentConfig(seed=seed)
            print(f"\nRUN {run_idx}/{total_runs}: {rule_name} | seed={seed}")

            # 1. Generate training data
            teacher = create_teacher(config.input_dim, seed=config.seed)
            tutor = generate_teacher_experiences(
                teacher,
                config.num_tutor_examples,
                1.0 / config.tutor_snr,
                config.seed + 1,
                "tutor",
            )
            practice = generate_teacher_experiences(
                teacher,
                config.num_practice_examples,
                1.0 / config.practice_snr,
                config.seed + 2,
                "practice",
            )
            train_x = np.vstack((tutor.x, practice.x))
            train_y = np.concatenate((tutor.y, practice.y))

            # 2. Generate validation data
            evaluation_noise = 0.0 if np.isinf(config.evaluation_snr) else 1.0 / config.evaluation_snr
            validation = generate_teacher_experiences(
                teacher,
                config.num_validation_examples,
                evaluation_noise,
                config.seed + 3,
                "evaluation",
            )
            validation_x, validation_y = validation.x, validation.y

            # 3. Create DeepStudent
            student = DeepStudent(
                input_dim=config.input_dim,
                cortex_dim=config.hidden_dim,
                readout_dim=50,
                seed=config.seed,
            )

            # 4. Create Notebook
            notebook = SparseHopfieldNotebook(
                notebook_dim=config.notebook_dim,
                sparsity=config.notebook_sparsity,
                seed=config.seed,
            )
            notebook.encode_batch(x=train_x, y=train_y)

            # 5. Create Learning Rule
            learning_rule = create_learning_rule(rule_name, config)

            # 6. Create Replay System
            replay = SleepReplay(
                student=student,
                notebook=notebook,
                learning_rule=learning_rule,
                replay_cycles=config.replay_cycles,
                replays_per_epoch=config.replays_per_epoch,
                update_w2=config.update_w2,
                replay_mode=config.replay_mode,
            )

            # 7. Create Go-CLS Controller
            controller = GoCLSController(
                replay=replay,
                patience=config.patience,
                min_delta=config.min_delta,
                max_epochs=config.max_epochs,
            )

            initial_val_loss = controller.validation_loss(student, validation_x, validation_y)

            # 8. Run Consolidation
            consolidation = controller.run(
                x_validation=validation_x,
                y_validation=validation_y,
            )

            # 9. Loss histories
            replay_losses = np.asarray([m.loss for m in consolidation.history], dtype=float)
            val_losses = np.asarray(consolidation.validation_losses, dtype=float)

            for epoch_idx, (r_loss, v_loss) in enumerate(zip(replay_losses, val_losses), start=1):
                history_records.append({
                    "learning_rule": rule_name,
                    "seed": seed,
                    "epoch": epoch_idx,
                    "replay_loss": r_loss,
                    "validation_loss": v_loss,
                })

            # 10. Stability calculation
            min_stability_epochs = int(np.ceil(config.stability_windows / config.stability_fraction))
            if replay_losses.size < min_stability_epochs:
                stability_val = np.nan
            else:
                stability = calculate_windowed_stability(
                    loss_history=replay_losses,
                    final_fraction=config.stability_fraction,
                    num_windows=config.stability_windows,
                )
                stability_val = stability.variance if hasattr(stability, "variance") else stability

            improvement_pct = (
                (initial_val_loss - consolidation.best_validation_loss) / initial_val_loss * 100.0
            )

            summary_records.append({
                "learning_rule": rule_name,
                "seed": seed,
                "epochs_executed": consolidation.epochs_executed,
                "initial_validation_loss": initial_val_loss,
                "best_validation_loss": consolidation.best_validation_loss,
                "best_epoch": consolidation.best_epoch,
                "relative_improvement_pct": improvement_pct,
                "stability_variance": stability_val,
                "stop_reason": consolidation.stop_reason,
            })

            print(
                f"Done: {consolidation.epochs_executed} epochs | "
                f"Best Val Loss: {consolidation.best_validation_loss:.4f} | "
                f"Improvement: {improvement_pct:.2f}% | "
                f"Stop: {consolidation.stop_reason}"
            )
            run_idx += 1

    pd.DataFrame(summary_records).to_csv(RESULTS_DIR / "rq2_summary.csv", index=False)
    pd.DataFrame(history_records).to_csv(RESULTS_DIR / "rq2_histories.csv", index=False)
    print("\nRQ2 complete. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    run_rq2()