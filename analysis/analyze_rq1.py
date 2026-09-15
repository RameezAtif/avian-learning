"""
RQ1 Statistical and Graphical Analysis
======================================

Automatically analyses RQ1 experiment results.

Expected input:

    results/
        rq1_baseline/
            summary.csv
            histories.csv

Output:

    results/
        rq1_baseline/
            analysis/
                tables/
                    descriptive_statistics.csv
                    improvement_statistics.csv
                    acquisition_statistics.csv
                    stopping_statistics.csv
                    stability_statistics.csv
                    statistical_tests.csv

                figures/
                    validation_learning_curves.png
                    replay_learning_curves.png
                    normalized_validation_improvement.png
                    best_validation_loss.png
                    relative_validation_improvement.png
                    epochs_executed.png
                    stability_variance.png
                    acquisition_success.png

                analysis_report.txt

Run from the project root:

    python analysis/analyze_rq1.py

The script is designed to work automatically when more seeds/runs
are added to summary.csv and histories.csv.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "rq1_baseline"

SUMMARY_FILE = RESULTS_DIR / "rq1_summary.csv"
HISTORIES_FILE = RESULTS_DIR / "rq1_histories.csv"

ANALYSIS_DIR = RESULTS_DIR / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"

# Primary RQ1 rules
PRIMARY_RULES = [
    "gradient_descent",
    "chl",
    "qpc",
    "hebbian",
]

# Additional exploratory rule
EXPLORATORY_RULES = [
    "anti_hebbian",
]

ALL_RULES = PRIMARY_RULES + EXPLORATORY_RULES

RULE_LABELS = {
    "gradient_descent": "Gradient Descent",
    "chl": "CHL",
    "qpc": "QPC",
    "hebbian": "Pure Hebbian",
    "anti_hebbian": "Anti-Hebbian",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rule_label(rule):
    """Convert internal rule name into a readable label."""
    return RULE_LABELS.get(rule, rule)


def ensure_directories():
    """Create output directories if they do not exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load summary and history CSV files."""
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Could not find summary file:\n{SUMMARY_FILE}"
        )

    if not HISTORIES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find histories file:\n{HISTORIES_FILE}"
        )

    summary = pd.read_csv(SUMMARY_FILE)
    histories = pd.read_csv(HISTORIES_FILE)

    required_summary_columns = [
        "learning_rule",
        "seed",
        "initial_validation_loss",
        "best_validation_loss",
        "best_epoch",
        "acquisition_epoch",
        "acquisition_threshold",
        "stability",
        "stop_reason",
        "epochs_executed",
    ]

    required_history_columns = [
        "learning_rule",
        "seed",
        "epoch",
        "replay_loss",
        "validation_loss",
    ]

    missing_summary = [
        c for c in required_summary_columns
        if c not in summary.columns
    ]

    missing_history = [
        c for c in required_history_columns
        if c not in histories.columns
    ]

    if missing_summary:
        raise ValueError(
            "summary.csv is missing columns:\n"
            + "\n".join(missing_summary)
        )

    if missing_history:
        raise ValueError(
            "histories.csv is missing columns:\n"
            + "\n".join(missing_history)
        )

    return summary, histories


def add_readable_rule_names(df):
    """Add a human-readable learning-rule column."""
    df = df.copy()

    df["learning_rule_label"] = df["learning_rule"].map(
        rule_label
    )

    return df


def calculate_derived_metrics(summary):
    """
    Calculate metrics that are not directly stored in summary.csv.
    """

    df = summary.copy()

    # Absolute improvement in validation loss
    df["absolute_validation_improvement"] = (
        df["initial_validation_loss"]
        - df["best_validation_loss"]
    )

    # Relative improvement as a fraction
    df["relative_validation_improvement"] = (
        df["absolute_validation_improvement"]
        / df["initial_validation_loss"]
    )

    # Percentage improvement
    df["relative_validation_improvement_percent"] = (
        df["relative_validation_improvement"] * 100
    )

    # Whether acquisition was achieved
    df["acquisition_success"] = (
        df["acquisition_epoch"].notna()
    )

    # Whether the run stopped because of validation patience
    df["patience_stopped"] = (
        df["stop_reason"] == "validation_patience_exceeded"
    )

    # Whether maximum epochs were reached
    df["max_epochs_reached"] = (
        df["stop_reason"] == "max_epochs_reached"
    )

    df["learning_rule_label"] = df["learning_rule"].map(
        rule_label
    )

    return df


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(summary):
    """
    Calculate mean, standard deviation, median, minimum and
    maximum for the main performance metrics.
    """

    metrics = [
        "initial_validation_loss",
        "best_validation_loss",
        "absolute_validation_improvement",
        "relative_validation_improvement_percent",
        "epochs_executed",
    ]

    rows = []

    for rule in summary["learning_rule"].unique():

        subset = summary[
            summary["learning_rule"] == rule
        ]

        row = {
            "learning_rule": rule,
            "learning_rule_label": rule_label(rule),
            "n_runs": len(subset),
        }

        for metric in metrics:

            values = pd.to_numeric(
                subset[metric],
                errors="coerce"
            ).dropna()

            if len(values) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_min"] = np.nan
                row[f"{metric}_max"] = np.nan
            else:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1)
                row[f"{metric}_median"] = values.median()
                row[f"{metric}_min"] = values.min()
                row[f"{metric}_max"] = values.max()

        rows.append(row)

    result = pd.DataFrame(rows)

    result.to_csv(
        TABLES_DIR / "descriptive_statistics.csv",
        index=False
    )

    return result


# ============================================================
# VALIDATION IMPROVEMENT
# ============================================================

def improvement_statistics(summary):
    """Summarise validation-loss improvement."""

    rows = []

    for rule in summary["learning_rule"].unique():

        subset = summary[
            summary["learning_rule"] == rule
        ]

        absolute = subset[
            "absolute_validation_improvement"
        ].dropna()

        relative = subset[
            "relative_validation_improvement_percent"
        ].dropna()

        rows.append({
            "learning_rule": rule,
            "learning_rule_label": rule_label(rule),
            "n_runs": len(subset),

            "mean_absolute_improvement":
                absolute.mean(),

            "std_absolute_improvement":
                absolute.std(ddof=1),

            "mean_relative_improvement_percent":
                relative.mean(),

            "std_relative_improvement_percent":
                relative.std(ddof=1),

            "median_relative_improvement_percent":
                relative.median(),

            "min_relative_improvement_percent":
                relative.min(),

            "max_relative_improvement_percent":
                relative.max(),

            "runs_with_positive_improvement":
                (absolute > 0).sum(),

            "runs_with_negative_improvement":
                (absolute < 0).sum(),

        })

    result = pd.DataFrame(rows)

    result.to_csv(
        TABLES_DIR / "improvement_statistics.csv",
        index=False
    )

    return result


# ============================================================
# ACQUISITION
# ============================================================

def acquisition_statistics(summary):
    """Analyse acquisition success."""

    rows = []

    for rule in summary["learning_rule"].unique():

        subset = summary[
            summary["learning_rule"] == rule
        ]

        successful = subset[
            "acquisition_success"
        ]

        acquisition_epochs = subset.loc[
            successful,
            "acquisition_epoch"
        ].dropna()

        rows.append({
            "learning_rule": rule,
            "learning_rule_label": rule_label(rule),
            "n_runs": len(subset),

            "acquisition_successes":
                int(successful.sum()),

            "acquisition_success_rate":
                successful.mean(),

            "mean_acquisition_epoch":
                acquisition_epochs.mean()
                if len(acquisition_epochs) > 0
                else np.nan,

            "std_acquisition_epoch":
                acquisition_epochs.std(ddof=1)
                if len(acquisition_epochs) > 1
                else np.nan,

        })

    result = pd.DataFrame(rows)

    result.to_csv(
        TABLES_DIR / "acquisition_statistics.csv",
        index=False
    )

    return result


# ============================================================
# STOPPING BEHAVIOUR
# ============================================================

def stopping_statistics(summary):
    """Analyse why runs stopped."""

    rows = []

    for rule in summary["learning_rule"].unique():

        subset = summary[
            summary["learning_rule"] == rule
        ]

        total = len(subset)

        patience_count = (
            subset["stop_reason"]
            == "validation_patience_exceeded"
        ).sum()

        max_epoch_count = (
            subset["stop_reason"]
            == "max_epochs_reached"
        ).sum()

        other_count = (
            total
            - patience_count
            - max_epoch_count
        )

        rows.append({
            "learning_rule": rule,
            "learning_rule_label": rule_label(rule),
            "n_runs": total,

            "mean_epochs_executed":
                subset["epochs_executed"].mean(),

            "std_epochs_executed":
                subset["epochs_executed"].std(ddof=1),

            "patience_stopped":
                patience_count,

            "patience_stop_rate":
                patience_count / total,

            "max_epochs_reached":
                max_epoch_count,

            "max_epochs_rate":
                max_epoch_count / total,

            "other_stop_reason":
                other_count,

            "other_stop_rate":
                other_count / total,
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        TABLES_DIR / "stopping_statistics.csv",
        index=False
    )

    return result


# ============================================================
# STABILITY
# ============================================================

def stability_statistics(summary):
    """Analyse the stability metric."""

    rows = []

    for rule in summary["learning_rule"].unique():

        subset = summary[
            summary["learning_rule"] == rule
        ]

        values = pd.to_numeric(
            subset["stability"],
            errors="coerce"
        ).dropna()

        rows.append({
            "learning_rule": rule,
            "learning_rule_label": rule_label(rule),
            "n_runs": len(values),

            "mean_stability_variance":
                values.mean()
                if len(values) > 0
                else np.nan,

            "std_stability_variance":
                values.std(ddof=1)
                if len(values) > 1
                else np.nan,

            "median_stability_variance":
                values.median()
                if len(values) > 0
                else np.nan,

            "min_stability_variance":
                values.min()
                if len(values) > 0
                else np.nan,

            "max_stability_variance":
                values.max()
                if len(values) > 0
                else np.nan,

        })

    result = pd.DataFrame(rows)

    result.to_csv(
        TABLES_DIR / "stability_statistics.csv",
        index=False
    )

    return result


# ============================================================
# STATISTICAL TESTS
# ============================================================

def paired_tests(summary):
    """
    Perform paired statistical comparisons using seeds.

    This is especially important because the same seeds are used
    across learning rules.

    For each pair of rules:

        H0:
            The paired performance differences have median 0.

    We use the Wilcoxon signed-rank test when possible.

    We also report the paired mean difference.
    """

    rows = []

    rules = list(
        summary["learning_rule"].unique()
    )

    for i in range(len(rules)):

        for j in range(i + 1, len(rules)):

            rule_a = rules[i]
            rule_b = rules[j]

            a = summary[
                summary["learning_rule"] == rule_a
            ][[
                "seed",
                "best_validation_loss",
                "relative_validation_improvement_percent",
            ]]

            b = summary[
                summary["learning_rule"] == rule_b
            ][[
                "seed",
                "best_validation_loss",
                "relative_validation_improvement_percent",
            ]]

            merged = pd.merge(
                a,
                b,
                on="seed",
                suffixes=("_a", "_b")
            )

            if len(merged) < 2:
                continue

            # ------------------------------------------------
            # Best validation loss
            # ------------------------------------------------

            diff_best = (
                merged["best_validation_loss_a"]
                - merged["best_validation_loss_b"]
            )

            try:
                wilcoxon_best = stats.wilcoxon(
                    diff_best
                )

                p_best = wilcoxon_best.pvalue

            except ValueError:
                p_best = np.nan

            # ------------------------------------------------
            # Relative improvement
            # ------------------------------------------------

            diff_improvement = (
                merged[
                    "relative_validation_improvement_percent_a"
                ]
                -
                merged[
                    "relative_validation_improvement_percent_b"
                ]
            )

            try:
                wilcoxon_improvement = stats.wilcoxon(
                    diff_improvement
                )

                p_improvement = (
                    wilcoxon_improvement.pvalue
                )

            except ValueError:
                p_improvement = np.nan

            rows.append({
                "rule_a": rule_a,
                "rule_b": rule_b,

                "rule_a_label":
                    rule_label(rule_a),

                "rule_b_label":
                    rule_label(rule_b),

                "n_paired_runs":
                    len(merged),

                "mean_best_loss_difference_a_minus_b":
                    diff_best.mean(),

                "wilcoxon_p_best_validation_loss":
                    p_best,

                "mean_relative_improvement_difference_a_minus_b":
                    diff_improvement.mean(),

                "wilcoxon_p_relative_improvement":
                    p_improvement,

            })

    result = pd.DataFrame(rows)

    # Bonferroni correction
    if not result.empty:

        result["p_best_loss_bonferroni"] = np.minimum(
            result["wilcoxon_p_best_validation_loss"]
            * len(result),
            1.0
        )

        result["p_improvement_bonferroni"] = np.minimum(
            result["wilcoxon_p_relative_improvement"]
            * len(result),
            1.0
        )

    result.to_csv(
        TABLES_DIR / "statistical_tests.csv",
        index=False
    )

    return result


# ============================================================
# HISTORY PROCESSING
# ============================================================

def prepare_history(histories):
    """Prepare histories for plotting."""

    histories = histories.copy()

    histories["learning_rule_label"] = (
        histories["learning_rule"].map(
            rule_label
        )
    )

    return histories


def calculate_history_summary(histories):
    """
    Calculate mean and standard error of validation/replay loss
    at every epoch.
    """

    grouped = (
        histories
        .groupby(
            ["learning_rule", "epoch"],
            as_index=False
        )
        .agg(
            validation_loss_mean=(
                "validation_loss",
                "mean"
            ),

            validation_loss_std=(
                "validation_loss",
                "std"
            ),

            validation_loss_n=(
                "validation_loss",
                "count"
            ),

            replay_loss_mean=(
                "replay_loss",
                "mean"
            ),

            replay_loss_std=(
                "replay_loss",
                "std"
            ),

            replay_loss_n=(
                "replay_loss",
                "count"
            ),
        )
    )

    grouped["validation_loss_sem"] = (
        grouped["validation_loss_std"]
        / np.sqrt(grouped["validation_loss_n"])
    )

    grouped["replay_loss_sem"] = (
        grouped["replay_loss_std"]
        / np.sqrt(grouped["replay_loss_n"])
    )

    grouped["learning_rule_label"] = (
        grouped["learning_rule"].map(
            rule_label
        )
    )

    return grouped


# ============================================================
# PLOT HELPERS
# ============================================================

def save_figure(filename):
    """Save the current matplotlib figure."""
    path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved figure: {path}")


# ============================================================
# PLOT 1 — VALIDATION LEARNING CURVES
# ============================================================

def aggregate_history(histories, metric):
    """
    Aggregate a history metric across seeds for each learning rule and epoch.
    """
    return (
        histories
        .groupby(["learning_rule", "epoch"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def plot_validation_learning_curves(histories, output_dir):
    """
    Mean validation loss across seeds, with ±1 standard deviation.
    """
    plt.figure(figsize=(10, 6))

    aggregated = aggregate_history(histories, "validation_loss")

    for rule in ALL_RULES:
        data = aggregated[aggregated["learning_rule"] == rule]

        if data.empty:
            continue

        x = data["epoch"].to_numpy()
        mean = data["mean"].to_numpy()
        std = data["std"].fillna(0).to_numpy()

        plt.plot(x, mean, label=rule)
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "validation_learning_curves.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved figure: {path}")


def plot_replay_learning_curves(histories, output_dir):
    """
    Mean replay loss across seeds, with ±1 standard deviation.
    """
    plt.figure(figsize=(10, 6))

    aggregated = aggregate_history(histories, "replay_loss")

    for rule in ALL_RULES:
        data = aggregated[aggregated["learning_rule"] == rule]

        if data.empty:
            continue

        x = data["epoch"].to_numpy()
        mean = data["mean"].to_numpy()
        std = data["std"].fillna(0).to_numpy()

        plt.plot(x, mean, label=rule)
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Replay Loss")
    plt.title("Replay Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "replay_learning_curves.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved figure: {path}")


def plot_normalized_validation_improvement(summary, output_dir):
    """
    Relative validation-loss improvement from initial to best validation loss.
    """
    plt.figure(figsize=(10, 6))

    data = summary.copy()

    data["relative_improvement"] = (
        (data["initial_validation_loss"] - data["best_validation_loss"])
        / data["initial_validation_loss"]
        * 100
    )

    for rule in ALL_RULES:
        rule_data = data[data["learning_rule"] == rule]

        if rule_data.empty:
            continue

        x = np.arange(len(rule_data))

        plt.plot(
            x,
            rule_data["relative_improvement"].to_numpy(),
            marker="o",
            label=rule,
        )

    plt.axhline(0, linewidth=1)

    plt.xlabel("Run")
    plt.ylabel("Validation Loss Improvement (%)")
    plt.title("Relative Validation Loss Improvement")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "normalized_validation_improvement.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved figure: {path}")


def plot_best_validation_loss(summary, output_dir):
    """
    Distribution of best validation loss across seeds.
    """
    plt.figure(figsize=(10, 6))

    data = []
    labels = []

    for rule in ALL_RULES:
        values = summary.loc[
            summary["learning_rule"] == rule,
            "best_validation_loss",
        ].dropna().to_numpy()

        if len(values) > 0:
            data.append(values)
            labels.append(rule_label(rule))

    if not data:
        plt.close()
        return

    plt.boxplot(
        data,
        tick_labels=labels,
    )

    plt.xlabel("Learning Rule")
    plt.ylabel("Best Validation Loss")
    plt.title("Best Validation Loss by Learning Rule")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "best_validation_loss.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved figure: {path}")


def plot_acquisition_epoch(summary, output_dir):
    """
    Epoch at which the acquisition criterion was reached.

    If no run reaches the acquisition criterion, explicitly
    indicate this rather than producing a visually blank plot.
    """
    plt.figure(figsize=(10, 6))

    data = []
    labels = []

    for rule in ALL_RULES:
        values = pd.to_numeric(
            summary.loc[
                summary["learning_rule"] == rule,
                "acquisition_epoch",
            ],
            errors="coerce",
        ).dropna().to_numpy()

        if len(values) > 0:
            data.append(values)
            labels.append(rule_label(rule))

    if data:
        plt.boxplot(
            data,
            tick_labels=labels,
        )

        plt.ylabel("Acquisition Epoch")

    else:
        plt.text(
            0.5,
            0.5,
            "No runs reached the acquisition criterion",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )

        plt.xticks([])
        plt.ylabel("Acquisition Epoch")

    plt.xlabel("Learning Rule")
    plt.title("Acquisition Epoch by Learning Rule")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "acquisition_epoch.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved figure: {path}")


def plot_stability(summary, output_dir):
    """
    Stability variance across runs.
    """
    plt.figure(figsize=(10, 6))

    data = []
    labels = []

    for rule in ALL_RULES:
        values = summary.loc[
            summary["learning_rule"] == rule,
            "stability",
        ].dropna().to_numpy()

        if len(values) > 0:
            data.append(values)
            labels.append(rule_label(rule))

    if data:
        plt.boxplot(
            data,
            tick_labels=labels,
        )

    plt.xlabel("Learning Rule")
    plt.ylabel("Stability Variance")
    plt.title("Stability Variance by Learning Rule")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "stability_variance.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved figure: {path}")


def plot_epochs_executed(summary, output_dir):
    """
    Number of epochs executed before stopping.
    """
    plt.figure(figsize=(10, 6))

    data = []
    labels = []

    for rule in ALL_RULES:
        values = summary.loc[
            summary["learning_rule"] == rule,
            "epochs_executed",
        ].dropna().to_numpy()

        if len(values) > 0:
            data.append(values)
            labels.append(rule_label(rule))

    if data:
        plt.boxplot(
            data,
            tick_labels=labels,
        )

    plt.xlabel("Learning Rule")
    plt.ylabel("Epochs Executed")
    plt.title("Training Duration by Learning Rule")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "epochs_executed.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved figure: {path}")


def plot_best_epoch(summary, output_dir):
    """
    Epoch at which the best validation loss occurred.
    """
    plt.figure(figsize=(10, 6))

    data = []
    labels = []

    for rule in ALL_RULES:
        values = summary.loc[
            summary["learning_rule"] == rule,
            "best_epoch",
        ].dropna().to_numpy()

        if len(values) > 0:
            data.append(values)
            labels.append(rule_label(rule))

    if data:
        plt.boxplot(
            data,
            tick_labels=labels,
        )

    plt.xlabel("Learning Rule")
    plt.ylabel("Best Epoch")
    plt.title("Best Validation Epoch by Learning Rule")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "best_epoch.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved figure: {path}")


# ============================================================
# TEXT REPORT
# ============================================================

def create_text_report(
    summary,
    descriptive,
    improvement,
    acquisition,
    stopping,
    stability,
    tests
):

    report_file = ANALYSIS_DIR / "analysis_report.txt"

    lines = []

    lines.append(
        "RQ1 STATISTICAL AND GRAPHICAL ANALYSIS"
    )

    lines.append(
        "=" * 50
    )

    lines.append("")

    lines.append(
        f"Number of runs: {len(summary)}"
    )

    lines.append(
        f"Learning rules: "
        f"{', '.join(summary['learning_rule'].unique())}"
    )

    lines.append("")

    # --------------------------------------------------------
    # MAIN SUMMARY
    # --------------------------------------------------------

    lines.append(
        "MAIN PERFORMANCE SUMMARY"
    )

    lines.append(
        "-" * 50
    )

    for _, row in improvement.iterrows():

        lines.append(
            f"{row['learning_rule_label']}: "
            f"mean relative improvement = "
            f"{row['mean_relative_improvement_percent']:.6f}% "
            f"(SD = "
            f"{row['std_relative_improvement_percent']:.6f}%)"
        )

    lines.append("")

    # --------------------------------------------------------
    # ACQUISITION
    # --------------------------------------------------------

    lines.append(
        "ACQUISITION"
    )

    lines.append(
        "-" * 50
    )

    for _, row in acquisition.iterrows():

        lines.append(
            f"{row['learning_rule_label']}: "
            f"{int(row['acquisition_successes'])}/"
            f"{int(row['n_runs'])} successful "
            f"({row['acquisition_success_rate'] * 100:.2f}%)"
        )

    lines.append("")

    # --------------------------------------------------------
    # STOPPING
    # --------------------------------------------------------

    lines.append(
        "STOPPING BEHAVIOUR"
    )

    lines.append(
        "-" * 50
    )

    for _, row in stopping.iterrows():

        lines.append(
            f"{row['learning_rule_label']}: "
            f"mean epochs = "
            f"{row['mean_epochs_executed']:.2f}, "
            f"patience stop rate = "
            f"{row['patience_stop_rate'] * 100:.2f}%, "
            f"max epoch rate = "
            f"{row['max_epochs_rate'] * 100:.2f}%"
        )

    lines.append("")

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    lines.append(
        "STABILITY"
    )

    lines.append(
        "-" * 50
    )

    for _, row in stability.iterrows():

        if pd.isna(
            row["mean_stability_variance"]
        ):
            continue

        lines.append(
            f"{row['learning_rule_label']}: "
            f"mean stability variance = "
            f"{row['mean_stability_variance']:.10f}"
        )

    lines.append("")

    # --------------------------------------------------------
    # PAIRED TESTS
    # --------------------------------------------------------

    lines.append(
        "PAIRED STATISTICAL TESTS"
    )

    lines.append(
        "-" * 50
    )

    if tests.empty:

        lines.append(
            "No statistical comparisons available."
        )

    else:

        for _, row in tests.iterrows():

            lines.append(
                f"{row['rule_a_label']} vs "
                f"{row['rule_b_label']}: "
                f"Wilcoxon p (best loss) = "
                f"{row['wilcoxon_p_best_validation_loss']:.6f}; "
                f"Bonferroni p = "
                f"{row['p_best_loss_bonferroni']:.6f}"
            )

            lines.append(
                f"    Wilcoxon p "
                f"(relative improvement) = "
                f"{row['wilcoxon_p_relative_improvement']:.6f}; "
                f"Bonferroni p = "
                f"{row['p_improvement_bonferroni']:.6f}"
            )

    lines.append("")

    lines.append(
        "IMPORTANT INTERPRETATION NOTE"
    )

    lines.append(
        "-" * 50
    )

    lines.append(
        "These statistical results describe differences under "
        "the current experimental configuration. They should "
        "not be interpreted as evidence that one learning rule "
        "is biologically superior without considering the "
        "architecture, learning-rule implementation, SNR "
        "conditions, and experimental limitations."
    )

    report_file.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        f"Saved report: {report_file}"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_console_summary(
    summary,
    improvement,
    acquisition,
    stopping,
    stability,
    tests
):

    print()
    print("=" * 70)
    print("RQ1 STATISTICAL ANALYSIS")
    print("=" * 70)

    print()
    print(
        f"Total runs: {len(summary)}"
    )

    print()

    print(
        "RELATIVE VALIDATION IMPROVEMENT"
    )

    print("-" * 70)

    for _, row in improvement.iterrows():

        print(
            f"{row['learning_rule_label']:<20} "
            f"mean = "
            f"{row['mean_relative_improvement_percent']:>10.6f}%   "
            f"SD = "
            f"{row['std_relative_improvement_percent']:>10.6f}%"
        )

    print()

    print(
        "ACQUISITION"
    )

    print("-" * 70)

    for _, row in acquisition.iterrows():

        print(
            f"{row['learning_rule_label']:<20} "
            f"{int(row['acquisition_successes'])}/"
            f"{int(row['n_runs'])} successful"
        )

    print()

    print(
        "STOPPING"
    )

    print("-" * 70)

    for _, row in stopping.iterrows():

        print(
            f"{row['learning_rule_label']:<20} "
            f"mean epochs = "
            f"{row['mean_epochs_executed']:.2f}   "
            f"patience = "
            f"{row['patience_stop_rate'] * 100:.1f}%   "
            f"max epochs = "
            f"{row['max_epochs_rate'] * 100:.1f}%"
        )

    print()

    print(
        "STABILITY"
    )

    print("-" * 70)

    for _, row in stability.iterrows():

        if pd.isna(
            row["mean_stability_variance"]
        ):
            print(
                f"{row['learning_rule_label']:<20} "
                f"not available"
            )

        else:

            print(
                f"{row['learning_rule_label']:<20} "
                f"mean variance = "
                f"{row['mean_stability_variance']:.10f}"
            )

    print()

    print(
        "STATISTICAL TESTS"
    )

    print("-" * 70)

    if tests.empty:

        print(
            "No paired statistical tests available."
        )

    else:

        for _, row in tests.iterrows():

            print(
                f"{row['rule_a_label']} vs "
                f"{row['rule_b_label']}: "
                f"p = "
                f"{row['wilcoxon_p_best_validation_loss']:.6f}"
            )

    print()
    print("=" * 70)
    print(
        f"All outputs saved under:\n{ANALYSIS_DIR}"
    )
    print("=" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning
    )

    print()
    print(
        "Loading RQ1 results..."
    )

    ensure_directories()

    summary, histories = load_data()

    summary = calculate_derived_metrics(
        summary
    )

    histories = prepare_history(
        histories
    )

    print(
        f"Loaded {len(summary)} completed runs."
    )

    print(
        f"Loaded {len(histories)} history rows."
    )

    # --------------------------------------------------------
    # TABLES
    # --------------------------------------------------------

    descriptive = descriptive_statistics(
        summary
    )

    improvement = improvement_statistics(
        summary
    )

    acquisition = acquisition_statistics(
        summary
    )

    stopping = stopping_statistics(
        summary
    )

    stability = stability_statistics(
        summary
    )

    tests = paired_tests(
        summary
    )

    history_summary = calculate_history_summary(
        histories
    )

    # Save derived run-level data as well
    summary.to_csv(
        TABLES_DIR / "derived_run_metrics.csv",
        index=False
    )

    history_summary.to_csv(
        TABLES_DIR / "history_statistics.csv",
        index=False
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    print()
    print(
        "Generating figures..."
    )

    figures_dir = Path("results/rq1_baseline/analysis/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)


    plot_validation_learning_curves(
        histories,
        figures_dir,
    )

    plot_replay_learning_curves(
        histories,
        figures_dir,
    )

    plot_normalized_validation_improvement(
        summary,
        figures_dir,
    )

    plot_best_validation_loss(
        summary,
        figures_dir,
    )

    plot_acquisition_epoch(
        summary,
        figures_dir,
    )

    plot_stability(
        summary,
        figures_dir,
    )

    plot_epochs_executed(
        summary,
        figures_dir,
    )

    plot_best_epoch(
        summary,
        figures_dir,
    )

    

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    create_text_report(
        summary,
        descriptive,
        improvement,
        acquisition,
        stopping,
        stability,
        tests
    )

    # --------------------------------------------------------
    # CONSOLE
    # --------------------------------------------------------

    print_console_summary(
        summary,
        improvement,
        acquisition,
        stopping,
        stability,
        tests
    )


if __name__ == "__main__":
    main()