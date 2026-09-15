from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for one complete experiment condition.
    """

    num_tutor_examples: int = 50
    num_practice_examples: int = 50
    num_validation_examples: int = 500

    input_dim: int = 100
    hidden_dim: int = 100

    tutor_snr: float = 20.0
    practice_snr: float = 3.0
    evaluation_snr: float = float("inf")

    notebook_dim: int = 2000
    notebook_sparsity: float = 0.05

    replay_cycles: int = 9
    replays_per_epoch: int = 64

    max_epochs: int = 300

    patience: int = 20
    min_delta: float = 1e-6

    learning_rate: float = 0.01

    acquisition_improvement_fraction: float = 0.50

    stability_fraction: float = 0.10
    stability_windows: int = 5

    update_w2: bool = True
    # Faithful episodic replay is the primary Go-CLS condition. ``hopfield``
    # is a retrieval-bias diagnostic, not a primary experimental condition.
    replay_mode: str = "stored"

    gradient_clip: float | None = 1.0

    seed: int = 42
