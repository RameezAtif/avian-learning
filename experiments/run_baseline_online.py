"""
Baseline A — Online learning.

Streaming mode: one data point at a time, train, discard.
No notebook, no replay, no Go-CLS gating.

Logs at cumulative gradient step counts so that early-step behaviour
is visible on a log x-axis, and the initial state (step 1) is captured
before any training.

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
MAX_EPOCHS = 1500


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


def _should_log(step):
    """Return True if we should record a snapshot at this step count."""
    if step <= 500:
        return True                            # dense early logging
    if step <= 5000:
        return step % 10 == 0                  # every 10 steps
    if step <= 50000:
        return step % 100 == 0                 # every 100 steps
    return step % 1000 == 0                    # every 1000 steps


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

    # Log initial state (before any training)
    initial_val_out = student.forward(val.x)
    initial_val_loss = mse(val.y, initial_val_out.y_hat)
    initial_train_out = student.forward(train_x[:1])
    initial_train_loss = mse(train_y[:1], initial_train_out.y_hat)

    log_steps = [1]
    train_losses = [initial_train_loss]
    val_losses = [initial_val_loss]

    cumulative_steps = 0
    epoch_loss_sum = 0.0
    n_seen = 0

    for epoch in range(config.max_epochs):
        rng = np.random.default_rng(seed * 1000 + epoch)
        order = rng.permutation(len(train_x))

        for idx in order:
            x_single = train_x[idx:idx + 1]
            y_single = train_y[idx:idx + 1]

            out = student.forward(x_single)
            loss = mse(y_single, out.y_hat)
            epoch_loss_sum += loss
            n_seen += 1
            cumulative_steps += 1

            upd = rule.calculate_update(
                x=x_single, y=y_single,
                h=out.h_ff, y_hat=out.y_hat,
                w1=student.W1, w2=student.W2,
            )
            student.W1 += upd.delta_w1
            if config.update_w2:
                student.W2 += upd.delta_w2

            if _should_log(cumulative_steps):
                val_out = student.forward(val.x)
                log_steps.append(cumulative_steps)
                train_losses.append(epoch_loss_sum / n_seen)
                val_losses.append(mse(val.y, val_out.y_hat))

    log_steps = np.array(log_steps)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    best_idx = int(np.argmin(val_losses))

    return {
        "learning_rule": rule_name,
        "seed": seed,
        "initial_validation_loss": float(initial_val_loss),
        "best_validation_loss": float(val_losses[best_idx]),
        "best_step": int(log_steps[best_idx]),
        "steps_executed": int(cumulative_steps),
        "relative_improvement_pct": 100.0 * (
            1 - val_losses[best_idx] / initial_val_loss
        ),
        "log_steps": log_steps,
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
                if k not in ("log_steps", "train_losses", "val_losses")
            })

            for i in range(len(result["log_steps"])):
                history_rows.append({
                    "learning_rule": rule,
                    "seed": seed,
                    "step": int(result["log_steps"][i]),
                    "train_loss": float(result["train_losses"][i]),
                    "validation_loss": float(result["val_losses"][i]),
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