from dataclasses import dataclass

import numpy as np

from consolidation.controller import (
    ConsolidationResult,
    GoCLSController,
)
from consolidation.replay import SleepReplay
from data.teacher import generate_teacher_dataset
from learning.gradient_descent import GradientDescentRule
from learning.plasticity import ContinuousPlasticityRule
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
    snr: float
    seed: int

    replay_losses: np.ndarray
    validation_losses: np.ndarray

    acquisition_epoch: int | None

    stability: StabilityResult 

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

        return ContinuousPlasticityRule(
            gamma=1.0,
            eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
        )

    if learning_rule_name == "qpc":

        return ContinuousPlasticityRule(
            gamma=-1.0,
            eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
        )

    if learning_rule_name == "hebbian":

        return ContinuousPlasticityRule(
            gamma=0.0,
            eta=0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
        )

    if learning_rule_name == "anti_hebbian":

        return ContinuousPlasticityRule(
            gamma=0.0,
            eta=-0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
        )

    raise ValueError(
        f"Unknown learning rule: {learning_rule_name}"
    )

def create_validation_dataset(
    x: np.ndarray,
    teacher_weights: np.ndarray,
    snr: float,
    seed: int,
):
    """
    Generate new validation inputs using the same teacher
    weights as the training dataset.
    """

    rng = np.random.default_rng(seed)

    input_dim = x.shape[1]

    if np.isinf(snr):
        noise_variance = 0.0
    else:
        noise_variance = 1.0 / (snr + 1.0)

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_variance),
        size=x.shape[0],
    )

    signal = x @ teacher_weights

    y = signal + noise

    return y

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

    train = generate_teacher_dataset(
        num_examples=config.num_train_examples,
        input_dim=config.input_dim,
        snr=config.snr,
        seed=config.seed,
    )

    # ---------------------------------------------------------
    # 2. Generate validation data using SAME teacher
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        config.seed + 1
    )

    validation_x = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(config.input_dim),
        size=(
            config.num_validation_examples,
            config.input_dim,
        ),
    )

    validation_y = create_validation_dataset(
        x=validation_x,
        teacher_weights=train.teacher_weights,
        snr=config.snr,
        seed=config.seed + 2,
    )

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
        x=train.x,
        y=train.y,
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
    # 10. Determine acquisition epoch
    # ---------------------------------------------------------

    acquisition_epoch = None

    below_threshold = np.where(
        replay_losses <= config.acquisition_epsilon
    )[0]

    if below_threshold.size > 0:
        acquisition_epoch = int(
            below_threshold[0] + 1
        )

    # ---------------------------------------------------------
    # 11. Calculate stability
    # ---------------------------------------------------------

    stability = calculate_windowed_stability(
        loss_history=replay_losses,
        final_fraction=config.stability_fraction,
        num_windows=config.stability_windows,
    )

    return ExperimentResult(
        learning_rule=learning_rule_name,
        snr=config.snr,
        seed=config.seed,
        replay_losses=replay_losses,
        validation_losses=validation_losses,
        acquisition_epoch=acquisition_epoch,
        stability=stability,
        best_validation_loss=(
            consolidation.best_validation_loss
        ),
        best_epoch=consolidation.best_epoch,
        epochs_executed=(
            consolidation.epochs_executed
        ),
    )