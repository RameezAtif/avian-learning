"""
Single-file experiment: Go-CLS with SNR-gated replay, at LOWER SNR.

This runs the same SNR-gated Go-CLS mechanism as before, but with noisy
data. The previous run used tutor_snr = 20, practice_snr = 3 — very
clean signal, so the gating mechanism had nothing to filter.

This run uses:
    tutor_snr    = 5   (moderately noisy)
    practice_snr = 1   (very noisy)

The contrast between predictable (tutor) and unpredictable (practice)
experiences is much larger in absolute terms. If the SNR-gating mechanism
is going to help anywhere, it should help here.

This file is standalone. It does not modify any existing code.

Produces:
    results/baseline_go_cls_snr_lownoise/go_cls_snr_summary.csv
    results/baseline_go_cls_snr_lownoise/go_cls_snr_histories.csv
    results/baseline_go_cls_snr_lownoise/figures/*.png
    results/baseline_go_cls_snr_lownoise/summary_table.csv
    results/baseline_go_cls_snr_lownoise/summary_table.tex
    results/baseline_go_cls_snr_lownoise/statistical_tests.csv

Run:
    python experiments/run_baseline_go_cls_snr_lownoise.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import replace

from data.teacher import create_teacher, generate_teacher_experiences
from experiments.config import ExperimentConfig
from learning.chl import ContrastiveHebbianRule
from learning.gradient_descent import GradientDescentRule
from learning.plasticity import ContinuousPlasticityRule
from models.student import Student


# =====================================================================
# Configuration
# =====================================================================

RULES = ["gradient_descent", "chl", "qpc", "hebbian", "anti_hebbian"]
LABELS = {
    "gradient_descent": "Gradient Descent",
    "chl": "CHL",
    "qpc": "QPC",
    "hebbian": "Pure Hebbian",
    "anti_hebbian": "Anti-Hebbian",
}
COLORS = {
    "gradient_descent": "tab:blue",
    "chl": "tab:orange",
    "qpc": "tab:green",
    "hebbian": "tab:red",
    "anti_hebbian": "tab:purple",
}
CONDITION_COLORS = {
    "CLS": "#CC6677",
    "Go-CLS (empirical)": "#228833",
    "Go-CLS (SNR-gated)": "#AA4499",
}

SEEDS = list(range(42, 72))
MAX_EPOCHS = 500
PATIENCE = 20
MIN_DELTA = 1e-6
BATCH_SIZE = 64

# Lower SNR — this is the change
TUTOR_SNR = 5.0
PRACTICE_SNR = 1.0

RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_go_cls_snr_lownoise"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Comparison baselines (existing high-SNR results)
EXISTING_CLS = PROJECT_ROOT / "results" / "baseline_cls" / "baseline_cls_summary.csv"
EXISTING_CLS_HIST = PROJECT_ROOT / "results" / "baseline_cls" / "baseline_cls_histories.csv"
EXISTING_EMPIRICAL = PROJECT_ROOT / "results" / "rq1_baseline" / "rq1_summary.csv"
EXISTING_EMPIRICAL_HIST = PROJECT_ROOT / "results" / "rq1_baseline" / "rq1_histories.csv"


# =====================================================================
# Rule factory
# =====================================================================

def create_rule(rule_name, config):
    if rule_name == "gradient_descent":
        return GradientDescentRule(
            learning_rate=config.learning_rate,
            update_w2=config.update_w2,
            gradient_clip=config.gradient_clip,
        )
    if rule_name == "chl":
        return ContrastiveHebbianRule(
            learning_rate=config.learning_rate,
            gamma=1.0, eta=0.0,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "qpc":
        return ContinuousPlasticityRule(
            gamma=-1.0, eta=0.0,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "hebbian":
        return ContinuousPlasticityRule(
            gamma=0.0, eta=0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    if rule_name == "anti_hebbian":
        return ContinuousPlasticityRule(
            gamma=0.0, eta=-0.1,
            learning_rate=config.learning_rate,
            gradient_clip=config.gradient_clip,
            update_w2=config.update_w2,
        )
    raise ValueError(f"Unknown rule: {rule_name}")


# =====================================================================
# SNR-gated sampling
# =====================================================================

def sample_snr_gated(snr_weights, batch_size, rng):
    return rng.choice(len(snr_weights), size=batch_size, p=snr_weights, replace=True)


def sample_uniform(n_samples, batch_size, rng):
    return rng.integers(0, n_samples, size=batch_size)


# =====================================================================
# Single run
# =====================================================================

def run_one(rule_name, seed, config, gating):
    """
    gating = "snr"    -> SNR-weighted replay
    gating = "uniform" -> uniform replay
    """
    # 1. Data
    teacher = create_teacher(config.input_dim, seed=seed)
    tutor = generate_teacher_experiences(
        teacher, config.num_tutor_examples,
        1.0 / TUTOR_SNR, seed + 1, "tutor",
    )
    practice = generate_teacher_experiences(
        teacher, config.num_practice_examples,
        1.0 / PRACTICE_SNR, seed + 2, "practice",
    )
    train_x = np.vstack((tutor.x, practice.x))
    train_y = np.concatenate((tutor.y, practice.y))
    n_train = len(train_x)

    # 2. SNR weights
    snr_values = np.concatenate([
        np.full(len(tutor.x), TUTOR_SNR),
        np.full(len(practice.x), PRACTICE_SNR),
    ])
    snr_weights = snr_values / snr_values.sum()

    # 3. Validation (noiseless, held out)
    eval_noise = (
        0.0 if np.isinf(config.evaluation_snr)
        else 1.0 / config.evaluation_snr
    )
    val = generate_teacher_experiences(
        teacher, config.num_validation_examples,
        eval_noise, seed + 3, "evaluation",
    )

    # 4. Student + rule
    student = Student(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        seed=seed,
    )
    rule = create_rule(rule_name, config)

    rng = np.random.default_rng(seed)

    # 5. Loop
    train_losses = []
    val_losses = []

    best_val = np.inf
    epochs_no_improve = 0
    stop_reason = "max_epochs_reached"

    for epoch in range(1, MAX_EPOCHS + 1):
        if gating == "snr":
            indices = sample_snr_gated(snr_weights, BATCH_SIZE, rng)
        else:
            indices = sample_uniform(n_train, BATCH_SIZE, rng)

        xb = train_x[indices]
        yb = train_y[indices]

        out = student.forward(xb)
        train_loss = 0.5 * float(np.mean((yb - out.y_hat) ** 2))

        update = rule.calculate_update(
            x=xb, y=yb,
            h=out.h_ff, y_hat=out.y_hat,
            w1=student.W1, w2=student.W2,
        )
        student.W1 += update.delta_w1
        if config.update_w2:
            student.W2 += update.delta_w2

        # Validation
        val_out = student.forward(val.x)
        val_loss = 0.5 * float(np.mean((val.y - val_out.y_hat) ** 2))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Patience stop
        if best_val - val_loss > MIN_DELTA:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            stop_reason = "validation_patience_exceeded"
            break

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    return {
        "learning_rule": rule_name,
        "seed": seed,
        "gating": gating,
        "initial_validation_loss": float(val_losses[0]),
        "final_validation_loss": float(val_losses[-1]),
        "final_training_loss": float(train_losses[-1]),
        "epochs_executed": len(train_losses),
        "stop_reason": stop_reason,
        "relative_improvement_pct": (
            100.0 * (1 - float(val_losses[-1]) / float(val_losses[0]))
        ),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# =====================================================================
# Run all experiments (both gating variants under low SNR)
# =====================================================================

def run_all():
    config = ExperimentConfig()
    summary_rows = []
    history_rows = []

    total = len(RULES) * len(SEEDS) * 2
    count = 0

    for rule in RULES:
        for seed in SEEDS:
            for gating in ("uniform", "snr"):
                count += 1
                tag = "SNR-gated" if gating == "snr" else "uniform"
                print(f"[{count}/{total}] {rule} seed={seed} ({tag})")

                result = run_one(rule, seed, config, gating)

                summary_rows.append({
                    k: v for k, v in result.items()
                    if k not in ("train_losses", "val_losses")
                })

                for ep in range(len(result["train_losses"])):
                    history_rows.append({
                        "learning_rule": rule,
                        "seed": seed,
                        "gating": gating,
                        "epoch": ep + 1,
                        "train_loss": float(result["train_losses"][ep]),
                        "validation_loss": float(result["val_losses"][ep]),
                    })

    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / "go_cls_snr_summary.csv", index=False,
    )
    pd.DataFrame(history_rows).to_csv(
        RESULTS_DIR / "go_cls_snr_histories.csv", index=False,
    )
    print(f"\nSaved summary and histories to {RESULTS_DIR}")
    return pd.DataFrame(summary_rows)


# =====================================================================
# Figures
# =====================================================================

def smooth(x, window=8):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().values


def plot_curves(hist_df, metric, title, filename):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)

    for ax, rule in zip(axes, RULES):
        sub = hist_df[hist_df["learning_rule"] == rule]

        for gating, label, color in [
            ("uniform", "Low-noise uniform (CLS-like)", "#CC6677"),
            ("snr",     "Low-noise SNR-gated",          "#AA4499"),
        ]:
            gsub = sub[sub["gating"] == gating]
            if gsub.empty:
                continue
            agg = gsub.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
            mean = smooth(agg["mean"].values)
            std = agg["std"].fillna(0).values
            ax.plot(agg["epoch"], mean, label=label, color=color, linewidth=1.8)
            ax.fill_between(agg["epoch"], mean - std, mean + std,
                            alpha=0.15, color=color)

        ax.set_title(LABELS[rule])
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / filename}")


def plot_final_comparison(summary_df):
    """Bar chart: final validation loss for uniform vs SNR-gated at low SNR."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(RULES))
    width = 0.35

    uniform_means, snr_means = [], []
    uniform_stds, snr_stds = [], []

    for rule in RULES:
        u = summary_df[
            (summary_df["gating"] == "uniform")
            & (summary_df["learning_rule"] == rule)
        ]["final_validation_loss"]
        s = summary_df[
            (summary_df["gating"] == "snr")
            & (summary_df["learning_rule"] == rule)
        ]["final_validation_loss"]
        uniform_means.append(u.mean())
        uniform_stds.append(u.std(ddof=1))
        snr_means.append(s.mean())
        snr_stds.append(s.std(ddof=1))

    ax.bar(x - width/2, uniform_means, width, yerr=uniform_stds, capsize=3,
           label="Low-noise uniform", color="#CC6677", alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, snr_means, width, yerr=snr_stds, capsize=3,
           label="Low-noise SNR-gated", color="#AA4499", alpha=0.85,
           edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in RULES], rotation=15)
    ax.set_ylabel("Final validation loss (lower is better)")
    ax.set_title(
        f"Low-noise regime (tutor SNR = {TUTOR_SNR}, practice SNR = {PRACTICE_SNR})"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_validation_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'final_validation_comparison.png'}")


# =====================================================================
# Summary table
# =====================================================================

def build_summary_table(summary_df):
    rows = []
    for rule in RULES:
        u = summary_df[
            (summary_df["gating"] == "uniform")
            & (summary_df["learning_rule"] == rule)
        ]
        s = summary_df[
            (summary_df["gating"] == "snr")
            & (summary_df["learning_rule"] == rule)
        ]

        rows.append({
            "learning_rule": LABELS[rule],
            "uniform_val_mean": u["final_validation_loss"].mean(),
            "uniform_val_std": u["final_validation_loss"].std(ddof=1),
            "snr_val_mean": s["final_validation_loss"].mean(),
            "snr_val_std": s["final_validation_loss"].std(ddof=1),
            "delta": s["final_validation_loss"].mean()
                     - u["final_validation_loss"].mean(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)

    print()
    print("=" * 80)
    print(f"LOW-NOISE REGIME (tutor SNR = {TUTOR_SNR}, practice SNR = {PRACTICE_SNR})")
    print("=" * 80)
    print(df.to_string(index=False))

    # LaTeX
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Low-noise Go-CLS. Tutor SNR = {TUTOR_SNR}, "
        rf"practice SNR = {PRACTICE_SNR}. Lower is better.}}",
        r"\label{tab:go_cls_snr_lownoise}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Rule} & \textbf{Uniform} & \textbf{SNR-gated} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['learning_rule']} & "
            f"{row['uniform_val_mean']:.3f} $\\pm$ {row['uniform_val_std']:.3f} & "
            f"{row['snr_val_mean']:.3f} $\\pm$ {row['snr_val_std']:.3f} & "
            f"{row['delta']:+.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(RESULTS_DIR / "summary_table.tex", "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved LaTeX: {RESULTS_DIR / 'summary_table.tex'}")

    return df


# =====================================================================
# Statistical tests
# =====================================================================

def statistical_tests(summary_df):
    rows = []
    for rule in RULES:
        u = summary_df[
            (summary_df["gating"] == "uniform")
            & (summary_df["learning_rule"] == rule)
        ].set_index("seed")["final_validation_loss"]
        s = summary_df[
            (summary_df["gating"] == "snr")
            & (summary_df["learning_rule"] == rule)
        ].set_index("seed")["final_validation_loss"]

        common = u.index.intersection(s.index)
        if len(common) < 2:
            continue

        diffs = (s.loc[common] - u.loc[common]).values

        # Handle all-zero case
        if np.allclose(diffs, 0):
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(diffs)
            except ValueError:
                p = np.nan

        rows.append({
            "learning_rule": rule,
            "mean_diff_snr_minus_uniform": diffs.mean(),
            "n_pairs": len(common),
            "wilcoxon_p": p,
        })

    out = pd.DataFrame(rows)
    out["bonferroni_p"] = np.minimum(out["wilcoxon_p"] * len(out), 1.0)
    out.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    print()
    print("=" * 80)
    print("STATISTICAL TESTS: SNR-gated vs uniform, low-noise regime")
    print("=" * 80)
    print(out.to_string(index=False))
    return out


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 80)
    print(f"Go-CLS SNR-gated replay, LOW-NOISE regime")
    print(f"  tutor_snr    = {TUTOR_SNR}")
    print(f"  practice_snr = {PRACTICE_SNR}")
    print("=" * 80)
    print()

    # 1. Run both uniform and SNR-gated at low noise
    summary_df = run_all()

    # 2. Load histories for plotting
    hist_df = pd.read_csv(RESULTS_DIR / "go_cls_snr_histories.csv")

    # 3. Figures
    plot_curves(
        hist_df, "train_loss",
        f"Training loss (low noise: tutor SNR = {TUTOR_SNR}, practice SNR = {PRACTICE_SNR})",
        "training_curves.png",
    )
    plot_curves(
        hist_df, "validation_loss",
        f"Validation loss (low noise: tutor SNR = {TUTOR_SNR}, practice SNR = {PRACTICE_SNR})",
        "validation_curves.png",
    )
    plot_final_comparison(summary_df)

    # 4. Summary table
    build_summary_table(summary_df)

    # 5. Stats
    statistical_tests(summary_df)

    print()
    print(f"All outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()