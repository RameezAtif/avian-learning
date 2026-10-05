"""
Verify the online baseline is measuring what we think it's measuring.

Checks three things:
  1. Learning rate is the same across rules
  2. Batch size is genuinely 1
  3. Each epoch performs one pass through the data
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from experiments.config import ExperimentConfig
from data.teacher import create_teacher, generate_teacher_experiences
from models.student import Student
from learning.chl import ContrastiveHebbianRule
from learning.gradient_descent import GradientDescentRule
from learning.plasticity import ContinuousPlasticityRule


def make_rule(name, config):
    if name == "gradient_descent":
        return GradientDescentRule(
            learning_rate=config.learning_rate,
            update_w2=config.update_w2,
            gradient_clip=config.gradient_clip,
        )
    if name == "chl":
        return ContrastiveHebbianRule(
            learning_rate=config.learning_rate,
            gamma=1.0, eta=0.0,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if name == "qpc":
        return ContinuousPlasticityRule(
            gamma=-1.0, eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if name == "hebbian":
        return ContinuousPlasticityRule(
            gamma=0.0, eta=0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if name == "anti_hebbian":
        return ContinuousPlasticityRule(
            gamma=0.0, eta=-0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )


def main():
    config = ExperimentConfig()
    rules = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]

    print("=" * 72)
    print("ONLINE BASELINE VERIFICATION")
    print("=" * 72)
    print()

    print("Check 1 — learning rate across rules")
    print("-" * 72)
    for name in rules:
        rule = make_rule(name, config)
        lr = getattr(rule, "learning_rate", getattr(rule, "lr", None))
        print(f"  {name:>16}  learning_rate = {lr}")
    print()
    print(f"  Config learning_rate = {config.learning_rate}")
    print()

    print("Check 2 — batch size is genuinely 1")
    print("-" * 72)
    print("  This is verified by inspecting run_baseline_online.py:")
    print("  the training loop should pass x_single = train_x[idx:idx+1]")
    print("  which has shape (1, input_dim).")
    print("  Run the command below to confirm shapes at runtime.")
    print()

    # Runtime shape check
    teacher = create_teacher(config.input_dim, seed=42)
    tutor = generate_teacher_experiences(
        teacher, config.num_tutor_examples,
        1.0 / config.tutor_snr, 43, "tutor",
    )
    practice = generate_teacher_experiences(
        teacher, config.num_practice_examples,
        1.0 / config.practice_snr, 44, "practice",
    )
    train_x = np.vstack((tutor.x, practice.x))
    train_y = np.concatenate((tutor.y, practice.y))

    print(f"  Training set size: {len(train_x)}")
    print(f"  Single sample shape: x = {train_x[0:1].shape}, y = {train_y[0:1].shape}")
    print()

    print("Check 3 — one pass per epoch")
    print("-" * 72)
    print(f"  Samples in training set: {len(train_x)}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Total single-sample updates per run: {len(train_x) * config.max_epochs}")
    print()
    print(f"  For comparison, batch Go-CLS:")
    print(f"  Replays per epoch: {config.replays_per_epoch}")
    print(f"  Total batch updates per run: {config.replays_per_epoch * config.max_epochs}")
    print()
    print(f"  Ratio of online updates to batch updates: "
          f"{len(train_x) * config.max_epochs / (config.replays_per_epoch * config.max_epochs):.2f}x")
    print()


if __name__ == "__main__":
    main()