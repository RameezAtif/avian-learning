from dataclasses import dataclass

import numpy as np

from consolidation.controller import (
    ConsolidationResult,
    GoCLSController,
)
from consolidation.replay import SleepReplay
from data.teacher import create_teacher, generate_teacher_experiences
from learning.gradient_descent import GradientDescentRule
from learning.plasticity import ContinuousPlasticityRule
from learning.chl import ContrastiveHebbianRule 
from memory.notebook import SparseHopfieldNotebook
from metrics.stability import (
    StabilityResult,
    calculate_windowed_stability,
)
from models.student import Student

from experiments.config import ExperimentConfig


@dataclass
class ExperimentResult:
    """
    Stores the complete result of one experiment condition.
    """

    learning_rule: str
    tutor_snr: float
    practice_snr: float
    seed: int

    replay_losses: np.ndarray
    validation_losses: np.ndarray

    acquisition_epoch: int | None
    acquisition_threshold: float
    initial_validation_loss: float

    stability: StabilityResult | None

    stop_reason: str

    best_validation_loss: float

    best_epoch: int

    epochs_executed: int

def create_learning_rule(
    learning_rule_name: str,
    config: ExperimentConfig,
):
    """
    Create the requested learning rule.
    """

    if learning_rule_name == "gradient_descent":

        return GradientDescentRule(
            learning_rate=config.learning_rate,
            update_w2=config.update_w2,
            gradient_clip=config.gradient_clip,
        )

    if learning_rule_name == "chl":

        return ContrastiveHebbianRule(
            learning_rate=config.learning_rate,
            feedback_strength=1.0,          # alpha — start at 1.0, tune later
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

    raise ValueError(
        f"Unknown learning rule: {learning_rule_name}"
    )

def run_experiment(
    learning_rule_name: str,
    config: ExperimentConfig,
) -> ExperimentResult:
    """
    Run one complete experimental condition.
    """

    # ---------------------------------------------------------
    # 1. Generate training data
    # ---------------------------------------------------------

    teacher = create_teacher(config.input_dim, seed=config.seed)
    tutor = generate_teacher_experiences(teacher, config.num_tutor_examples,
        1.0 / config.tutor_snr, config.seed + 1, "tutor")
    practice = generate_teacher_experiences(teacher, config.num_practice_examples,
        1.0 / config.practice_snr, config.seed + 2, "practice")
    train_x, train_y = np.vstack((tutor.x, practice.x)), np.concatenate((tutor.y, practice.y))

    # ---------------------------------------------------------
    # 2. Generate validation data using SAME teacher
    # ---------------------------------------------------------

    evaluation_noise = 0.0 if np.isinf(config.evaluation_snr) else 1.0 / config.evaluation_snr
    validation = generate_teacher_experiences(teacher, config.num_validation_examples,
        evaluation_noise, config.seed + 3, "evaluation")
    validation_x, validation_y = validation.x, validation.y

    # ---------------------------------------------------------
    # 3. Create Student
    # ---------------------------------------------------------

    student = Student(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )

    # ---------------------------------------------------------
    # 4. Create Notebook
    # ---------------------------------------------------------

    notebook = SparseHopfieldNotebook(
        notebook_dim=config.notebook_dim,
        sparsity=config.notebook_sparsity,
        seed=config.seed,
    )

    notebook.encode_batch(
        x=train_x,
        y=train_y,
    )

    # ---------------------------------------------------------
    # 5. Create learning rule
    # ---------------------------------------------------------

    learning_rule = create_learning_rule(
        learning_rule_name=learning_rule_name,
        config=config,
    )

    # ---------------------------------------------------------
    # 6. Create replay system
    # ---------------------------------------------------------

    replay = SleepReplay(
        student=student,
        notebook=notebook,
        learning_rule=learning_rule,
        replay_cycles=config.replay_cycles,
        replays_per_epoch=config.replays_per_epoch,
        update_w2=config.update_w2,
        replay_mode=config.replay_mode,
    )

    # ---------------------------------------------------------
    # 7. Create Go-CLS controller
    # ---------------------------------------------------------

    controller = GoCLSController(
        replay=replay,
        patience=config.patience,
        min_delta=config.min_delta,
        max_epochs=config.max_epochs,
    )

    # ---------------------------------------------------------
    initial_validation_loss = controller.validation_loss(student, validation_x, validation_y)

    # 8. Run consolidation
    # ---------------------------------------------------------

    consolidation = controller.run(
        x_validation=validation_x,
        y_validation=validation_y,
    )

    # ---------------------------------------------------------
    # 9. Extract loss histories
    # ---------------------------------------------------------

    replay_losses = np.asarray(
        [
            metric.loss
            for metric in consolidation.history
        ],
        dtype=float,
    )

    validation_losses = np.asarray(
        consolidation.validation_losses,
        dtype=float,
    )

    # ---------------------------------------------------------
    # 10. Acquisition is computed in the analysis phase.
    #
    # The threshold depends on the best loss achieved across
    # all rules and seeds, so it cannot be defined per-run.
    # ---------------------------------------------------------

    acquisition_epoch = None
    acquisition_threshold = float("nan")

    # ---------------------------------------------------------
    # 11. Calculate stability
    # ---------------------------------------------------------

    minimum_stability_epochs = int(
    np.ceil(
        config.stability_windows
        / config.stability_fraction
    )
)

    if replay_losses.size < minimum_stability_epochs:
        stability = None
    else:
        stability = calculate_windowed_stability(
            loss_history=replay_losses,
            final_fraction=config.stability_fraction,
            num_windows=config.stability_windows,
        )

    return ExperimentResult(
        learning_rule=learning_rule_name,
        tutor_snr=config.tutor_snr,
        practice_snr=config.practice_snr,
        seed=config.seed,
        replay_losses=replay_losses,
        validation_losses=validation_losses,
        acquisition_epoch=acquisition_epoch,
        acquisition_threshold=acquisition_threshold,
        initial_validation_loss=initial_validation_loss,
        stability=stability,
        stop_reason=consolidation.stop_reason,
        best_validation_loss=(
            consolidation.best_validation_loss
        ),
        best_epoch=consolidation.best_epoch,
        epochs_executed=(
            consolidation.epochs_executed
        ),
    )
