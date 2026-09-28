"""
Evidence script for supervisor.

Runs GD and CHL on hierarchical data for a single seed and prints
whether they produce different results. Contrast with Gaussian data,
where the control experiment showed CHL = GD + variance term.

Run from project root:
    python hierarchical/experiments/compare_chl_gd.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from hierarchical.data.hierarchical_teacher import generate_hierarchical_data
from hierarchical.models.vector_student import VectorStudent
from hierarchical.learning.plasticity_vector import ContinuousPlasticityVector
from hierarchical.learning.chl_vector import ContrastiveHebbianVector
from hierarchical.memory.notebook import StoredNotebook
from hierarchical.replay.replay import train_student, evaluate_validation
from hierarchical.experiments.config import HierarchicalConfig


def run(rule_name, config, seed):
    data = generate_hierarchical_data(
        n_super=config.n_super,
        n_sub_per_super=config.n_sub_per_super,
        samples_per_sub=config.samples_per_sub,
        input_dim=config.input_dim,
        within_category_noise=config.within_category_noise,
        super_separation=config.super_separation,
        seed=seed,
    )

    n = len(data.x)
    n_val = n // 5
    val_x, val_y = data.x[:n_val], data.y[:n_val]
    train_x, train_y = data.x[n_val:], data.y[n_val:]

    student = VectorStudent(
        config.input_dim, config.hidden_dim, config.output_dim, seed,
    )
    notebook = StoredNotebook()
    notebook.encode_batch(train_x, train_y)

    if rule_name == "gd":
        rule = ContinuousPlasticityVector(
            gamma=1.0, eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    elif rule_name == "chl":
        rule = ContrastiveHebbianVector(
            learning_rate=config.learning_rate,
            gamma=1.0, eta=0.0,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    else:
        raise ValueError(rule_name)

    initial = evaluate_validation(student, val_x, val_y)
    train_student(
        student, notebook, rule,
        config.max_epochs, config.batch_size, seed + 1,
    )
    final = evaluate_validation(student, val_x, val_y)

    improvement = 100 * (1 - final / initial)
    return initial, final, improvement


def main():
    config = HierarchicalConfig()
    seeds = [42, 43, 44]

    print("=" * 68)
    print("CHL vs GD on HIERARCHICAL data (nested labels)")
    print("=" * 68)
    print()
    print(f"Label dim: {config.output_dim}  "
          f"(super: {config.n_super}, sub: {config.n_sub_per_super})")
    print(f"Noise: {config.within_category_noise}  "
          f"Super separation: {config.super_separation}")
    print()
    print(f"{'seed':>5}  {'rule':>4}  {'initial':>10}  {'final':>10}  {'impr%':>8}")
    print("-" * 50)

    for seed in seeds:
        for rule in ["gd", "chl"]:
            init, fin, imp = run(rule, config, seed)
            print(f"{seed:>5}  {rule:>4}  {init:>10.6f}  {fin:>10.6f}  {imp:>8.2f}")

    print()
    print("Gaussian control (from earlier):")
    print("  GD:       ~16% improvement")
    print("  CHL:      ~39% improvement")
    print("  Both identical because CHL = GD + variance matching.")
    print()
    print("If CHL and GD now differ on hierarchical data, the label")
    print("structure is activating the correlation term.")


if __name__ == "__main__":
    main()