"""
RQ1 on hierarchical data.

Runs all five learning rules on hierarchical data.
Compares to the Gaussian results.

Run from project root:
    python hierarchical/experiments/run_rq1_hier.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from hierarchical.data.hierarchical_teacher import generate_hierarchical_data
from hierarchical.models.vector_student import VectorStudent
from hierarchical.learning.plasticity_vector import ContinuousPlasticityVector
from hierarchical.learning.chl_vector import ContrastiveHebbianVector
from hierarchical.memory.notebook import StoredNotebook
from hierarchical.replay.replay import train_student, evaluate_validation
from hierarchical.experiments.config import HierarchicalConfig


# -----------------------------------------------------------------
# Learning rule factory
# -----------------------------------------------------------------

def create_rule(rule_name, config):
    if rule_name == "gradient_descent":
        return ContinuousPlasticityVector(
            gamma=1.0, eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "chl":
        return ContrastiveHebbianVector(
            learning_rate=config.learning_rate,
            gamma=1.0, eta=0.0,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "qpc":
        return ContinuousPlasticityVector(
            gamma=-1.0, eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "hebbian":
        return ContinuousPlasticityVector(
            gamma=0.0, eta=0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "anti_hebbian":
        return ContinuousPlasticityVector(
            gamma=0.0, eta=-0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    raise ValueError(f"Unknown rule: {rule_name}")


# -----------------------------------------------------------------
# Single run
# -----------------------------------------------------------------

def run_single(rule_name, seed, config):
    # Override seed
    config = type(config)(**{**config.__dict__, "seed": seed})

    # Generate hierarchical data
    data = generate_hierarchical_data(
        n_super=config.n_super,
        n_sub_per_super=config.n_sub_per_super,
        samples_per_sub=config.samples_per_sub,
        input_dim=config.input_dim,
        within_category_noise=config.within_category_noise,
        super_separation=config.super_separation,
        seed=seed,
    )

    # Split: 80% train, 20% validation
    n = len(data.x)
    n_val = n // 5
    val_x = data.x[:n_val]
    val_y = data.y[:n_val]
    train_x = data.x[n_val:]
    train_y = data.y[n_val:]

    # Build student and notebook
    student = VectorStudent(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        seed=seed,
    )

    notebook = StoredNotebook()
    notebook.encode_batch(train_x, train_y)

    rule = create_rule(rule_name, config)

    # Initial validation loss (before training)
    initial_val_loss = evaluate_validation(student, val_x, val_y)

    # Train
    history = train_student(
        student=student,
        notebook=notebook,
        learning_rule=rule,
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        seed=seed + 1,
    )

    # Final validation loss
    final_val_loss = evaluate_validation(student, val_x, val_y)

    # Also track best-ever validation loss by snapshotting periodically
    # (for this simple version, we use final val loss as best)
    return {
        "learning_rule": rule_name,
        "seed": seed,
        "initial_validation_loss": initial_val_loss,
        "best_validation_loss": final_val_loss,
        "relative_improvement_pct": 100.0 * (
            1 - final_val_loss / initial_val_loss
        ),
        "replay_losses": history["replay_losses"],
    }


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    config = HierarchicalConfig()

    rules = [
        "gradient_descent",
        "chl",
        "qpc",
        "hebbian",
        "anti_hebbian",
    ]
    seeds = list(range(42, 52))  # 10 seeds for speed

    results_dir = PROJECT_ROOT / "hierarchical" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    history_rows = []

    total = len(rules) * len(seeds)
    count = 0

    for rule_name in rules:
        for seed in seeds:
            count += 1
            print(f"[{count}/{total}] {rule_name} seed={seed}")

            result = run_single(rule_name, seed, config)

            all_rows.append({
                "learning_rule": result["learning_rule"],
                "seed": result["seed"],
                "initial_validation_loss": result["initial_validation_loss"],
                "best_validation_loss": result["best_validation_loss"],
                "relative_improvement_pct": result["relative_improvement_pct"],
            })

            for epoch, loss in enumerate(result["replay_losses"]):
                history_rows.append({
                    "learning_rule": rule_name,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "replay_loss": loss,
                })

    summary_df = pd.DataFrame(all_rows)
    history_df = pd.DataFrame(history_rows)

    summary_df.to_csv(results_dir / "rq1_hier_summary.csv", index=False)
    history_df.to_csv(results_dir / "rq1_hier_histories.csv", index=False)

    print()
    print("=" * 68)
    print("HIERARCHICAL RQ1 — SUMMARY")
    print("=" * 68)
    print(summary_df.groupby("learning_rule")["relative_improvement_pct"].agg(
        ["mean", "std", "count"]
    ))
    print()
    print(f"Saved to: {results_dir}")


if __name__ == "__main__":
    main()