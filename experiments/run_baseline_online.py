"""
Baseline A — Online learning.

Streaming mode: one data point at a time, train, discard.
No notebook, no replay, no Go-CLS gating.

Run:
    python experiments/run_baseline_online.py
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.teacher import create_teacher, generate_teacher_experiences
from experiments.config import ExperimentConfig
from learning.chl import ContrastiveHebbianRule
from learning.gradient_descent import GradientDescentRule
from learning.plasticity import ContinuousPlasticityRule
from models.student import Student
from metrics.acquisition import compute_all_acquisition_epochs


RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
SEEDS = list(range(42, 72))
RESULTS = PROJECT_ROOT / "results" / "baseline_online"
RESULTS.mkdir(parents=True, exist_ok=True)


def create_rule(name, config):
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
    raise ValueError(name)


def run_one(rule_name, seed, config):
    # Data
    teacher = create_teacher(config.input_dim, seed=seed)
    tutor = generate_teacher_experiences(
        teacher, config.num_tutor_examples,
        1.0 / config.tutor_snr, seed + 1, "tutor",
    )
    practice = generate_teacher_experiences(
        teacher, config.num_practice_examples,
        1.0 / config.practice_snr, seed + 2, "practice",
    )
    train_x = np.vstack((tutor.x, practice.x))
    train_y = np.concatenate((tutor.y, practice.y))

    eval_noise = (
        0.0 if np.isinf(config.evaluation_snr)
        else 1.0 / config.evaluation_snr
    )
    val = generate_teacher_experiences(
        teacher, config.num_validation_examples,
        eval_noise, seed + 3, "evaluation",
    )

    student = Student(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        seed=seed,
    )
    rule = create_rule(rule_name, config)

    # Streaming: shuffle and present one at a time
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_x))

    val_losses = []
    initial_val = None

    for epoch in range(config.max_epochs):
        # One pass = one epoch through all data, one sample at a time
        for idx in order:
            x_single = train_x[idx:idx+1]      # shape (1, input_dim)
            y_single = train_y[idx:idx+1]      # shape (1,)

            out = student.forward(x_single)
            upd = rule.calculate_update(
                x=x_single, y=y_single,
                h=out.h_ff, y_hat=out.y_hat,
                w1=student.W1, w2=student.W2,
            )
            student.W1 += upd.delta_w1
            if config.update_w2:
                student.W2 += upd.delta_w2

        # Validate after each epoch
        val_out = student.forward(val.x)
        vloss = 0.5 * float(np.mean((val.y - val_out.y_hat) ** 2))
        if initial_val is None:
            initial_val = vloss
        val_losses.append(vloss)

    val_losses = np.array(val_losses)
    best_idx = int(np.argmin(val_losses))

    return {
        "learning_rule": rule_name,
        "seed": seed,
        "initial_validation_loss": initial_val,
        "best_validation_loss": float(val_losses[best_idx]),
        "best_epoch": best_idx + 1,
        "epochs_executed": len(val_losses),
        "relative_improvement_pct": 100.0 * (1 - val_losses[best_idx] / initial_val),
    }


def main():
    config = ExperimentConfig()
    rows = []
    history_rows = []

    total = len(RULES) * len(SEEDS)
    count = 0
    for rule in RULES:
        for seed in SEEDS:
            count += 1
            print(f"[{count}/{total}] {rule} seed={seed}")
            rows.append(run_one(rule, seed, config))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "baseline_online_summary.csv", index=False)

    print()
    print(df.groupby("learning_rule")["relative_improvement_pct"].agg(
        ["mean", "std", "count"]))
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()