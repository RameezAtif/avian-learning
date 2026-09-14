from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for one complete experiment condition.
    """

    num_train_examples: int = 1000
    num_validation_examples: int = 500

    input_dim: int = 100
    hidden_dim: int = 100

    snr: float = 4.0

    notebook_dim: int = 500
    notebook_sparsity: float = 0.05

    replay_cycles: int = 9
    replays_per_epoch: int = 32

    max_epochs: int = 100

    patience: int = 10
    min_delta: float = 1e-6

    learning_rate: float = 0.001

    acquisition_epsilon: float = 0.01

    stability_fraction: float = 0.10
    stability_windows: int = 5

    update_w2: bool = False

    gradient_clip: float | None = 1.0

    seed: int = 42