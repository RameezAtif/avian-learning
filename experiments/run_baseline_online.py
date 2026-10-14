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


RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
SEEDS = list(range(42, 72))

# MAX_EPOCHS = 5 for matched (500 updates), 500 for natural (50,000 updates)
MAX_EPOCHS = 5

if MAX_EPOCHS <= 10:
    RESULTS = PROJECT_ROOT / "results" / "baseline_online_matched"
else:
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


def mse(y, y_hat):
    return 0.5 * float(np.mean((y - y_hat) ** 2))


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

    val_losses = []
    train_losses = []
    initial_val = None

    for epoch in range(MAX_EPOCHS):
        rng = np.random.default_rng(seed * 1000 + epoch)
        order = rng.permutation(len(train_x))

        # Accumulate training loss across the epoch
        epoch_loss_sum = 0.0
        n_seen = 0

        for idx in order:
            x_single = train_x[idx:idx+1]
            y_single = train_y[idx:idx+1]

            out = student.forward(x_single)

            # Pre-update training loss
            loss = mse(y_single, out.y_hat)
            epoch_loss_sum += loss
            n_seen += 1

            upd = rule.calculate_update(
                x=x_single, y=y_single,
                h=out.h_ff, y_hat=out.y_hat,
                w1=student.W1, w2=student.W2,
            )
            student.W1 += upd.delta_w1
            if config.update_w2:
                student.W2 += upd.delta_w2

        train_losses.append(epoch_loss_sum / n_seen)

        # Validation
        val_out = student.forward(val.x)
        vloss = mse(val.y, val_out.y_hat)
        if initial_val is None:
            initial_val = vloss
        val_losses.append(vloss)

    val_losses = np.array(val_losses)
    train_losses = np.array(train_losses)
    best_idx = int(np.argmin(val_losses))

    return {
        "learning_rule": rule_name,
        "seed": seed,
        "initial_validation_loss": initial_val,
        "best_validation_loss": float(val_losses[best_idx]),
        "best_epoch": best_idx + 1,
        "epochs_executed": len(val_losses),
        "relative_improvement_pct": 100.0 * (
            1 - val_losses[best_idx] / initial_val
        ),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def main():
    config = ExperimentConfig()
    summary_rows = []
    history_rows = []

    total = len(RULES) * len(SEEDS)
    count = 0

    for rule in RULES:
        for seed in SEEDS:
            count += 1
            print(f"[{count}/{total}] {rule} seed={seed}")
            result = run_one(rule, seed, config)

            summary_rows.append({
                k: v for k, v in result.items()
                if k not in ("train_losses", "val_losses")
            })

            for ep in range(len(result["train_losses"])):
                history_rows.append({
                    "learning_rule": rule,
                    "seed": seed,
                    "epoch": ep + 1,
                    "train_loss": float(result["train_losses"][ep]),
                    "validation_loss": float(result["val_losses"][ep]),
                })

    pd.DataFrame(summary_rows).to_csv(
        RESULTS / "baseline_online_summary.csv", index=False,
    )
    pd.DataFrame(history_rows).to_csv(
        RESULTS / "baseline_online_histories.csv", index=False,
    )

    print()
    print(pd.DataFrame(summary_rows).groupby(
        "learning_rule")["relative_improvement_pct"].agg(
        ["mean", "std", "count"]))
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()