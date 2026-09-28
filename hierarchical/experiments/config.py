"""
Configuration for the hierarchical RQ1 experiment.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HierarchicalConfig:
    # Data
    n_super: int = 2
    n_sub_per_super: int = 10
    samples_per_sub: int = 15
    input_dim: int = 50
    hidden_dim: int = 50
    output_dim: int = 22            # = n_super + n_sub = 2 + 20
    within_category_noise: float = 1.5
    super_separation: float = 3.0

    # Training
    max_epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 0.01
    gradient_clip: float = 1.0
    update_w2: bool = True

    # Seeds
    seed: int = 42