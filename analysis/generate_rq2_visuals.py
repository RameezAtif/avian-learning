import sys
import ast
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RQ1_SUM_PATH = PROJECT_ROOT / "results" / "rq1_baseline" / "rq1_summary.csv"
RQ2_SUM_PATH = PROJECT_ROOT / "results" / "rq2_depth" / "rq2_summary.csv"
RQ1_HIST_PATH = PROJECT_ROOT / "results" / "rq1_baseline" / "rq1_histories.csv"
RQ2_HIST_PATH = PROJECT_ROOT / "results" / "rq2_depth" / "rq2_histories.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "rq2_depth" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RULE_MAPPING = {
    'gradient_descent': 'Gradient Descent',
    'chl': 'CHL',
    'hebbian': 'Pure Hebbian',
    'anti_hebbian': 'Anti-Hebbian',
    'qpc': 'QPC'
}

def parse_stability(val):
    """Extracts the float variance from the StabilityResult string if necessary."""
    if isinstance(val, str) and "stability_variance=" in val:
        try:
            return float(val.split("stability_variance=")[1].split(",")[0])
        except Exception:
            return np.nan
    return float(val)

def load_data():
    """Loads and merges summary and history data for RQ1 and RQ2."""
    df1_sum = pd.read_csv(RQ1_SUM_PATH)
    df1_sum['Architecture'] = 'Shallow (1-Layer)'
    df2_sum = pd.read_csv(RQ2_SUM_PATH)
    df2_sum['Architecture'] = 'Deep (2-Layer)'
    
    df_sum = pd.concat([df1_sum, df2_sum], ignore_index=True)
    df_sum['learning_rule'] = df_sum['learning_rule'].replace(RULE_MAPPING)
    df_sum['stability_variance'] = df_sum['stability_variance'].apply(parse_stability)

    df1_hist = pd.read_csv(RQ1_HIST_PATH)
    df1_hist['Architecture'] = 'Shallow (1-Layer)'
    df2_hist = pd.read_csv(RQ2_HIST_PATH)
    df2_hist['Architecture'] = 'Deep (2-Layer)'
    
    df_hist = pd.concat([df1_hist, df2_hist], ignore_index=True)
    df_hist['learning_rule'] = df_hist['learning_rule'].replace(RULE_MAPPING)

    return df_sum, df_hist

def plot_improvement_distribution(df_sum):
    """Graph 1: Box and Swarm plot showing the distribution of improvement across seeds."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    sns.boxplot(
        data=df_sum, x='learning_rule', y='relative_improvement_pct', hue='Architecture',
        palette=['#A0CBE8', '#F28E2B'], boxprops={'alpha': 0.6}, showfliers=False
    )
    sns.stripplot(
        data=df_sum, x='learning_rule', y='relative_improvement_pct', hue='Architecture',
        palette=['#4E79A7', '#E15759'], dodge=True, size=6, alpha=0.8, edgecolor='white', linewidth=1
    )
    
    plt.title('Distribution of Relative Validation Improvement (5 Seeds)', pad=15, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Relative Improvement (%)')
    
    # Fix legend duplication from combining boxplot and stripplot
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title='Architecture', loc='upper right')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_improvement_distribution.png", dpi=300)
    plt.close()

def plot_loss_trajectories(df_hist):
    """Graph 2: Facet grid of validation loss trajectories over time."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    
    g = sns.FacetGrid(
        df_hist, col="learning_rule", col_wrap=3, height=4, aspect=1.2, 
        sharey=False, despine=True
    )
    g.map_dataframe(
        sns.lineplot, x="epoch", y="validation_loss", hue="Architecture", 
        palette=['#4E79A7', '#E15759'], errorbar=('ci', 95), linewidth=2
    )
    
    g.set_axis_labels("Epoch", "Validation Loss (MSE)")
    g.set_titles(col_template="{col_name}", fontweight='bold')
    g.add_legend(title="Architecture")
    
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Speed of Acquisition: Validation Loss Trajectories by Learning Rule', fontweight='bold', fontsize=14)
    
    plt.savefig(OUTPUT_DIR / "02_loss_trajectories.png", dpi=300)
    plt.close()

def plot_stability(df_sum):
    """Graph 3: Point plot showing log-scaled stability variance."""
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    sns.pointplot(
        data=df_sum, x='learning_rule', y='stability_variance', hue='Architecture',
        palette=['#4E79A7', '#E15759'], markers=["o", "s"], capsize=.1, dodge=True
    )
    
    plt.yscale('log')
    plt.title('Representational Stability (Lower Variance = More Stable)', pad=15, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Windowed Variance (Log Scale)')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_stability_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Loading data and generating visualizations...")
    df_summary, df_history = load_data()
    
    plot_improvement_distribution(df_summary)
    print("- Generated 01_improvement_distribution.png")
    
    plot_loss_trajectories(df_history)
    print("- Generated 02_loss_trajectories.png")
    
    plot_stability(df_summary)
    print("- Generated 03_stability_comparison.png")
    
    print(f"All figures saved to: {OUTPUT_DIR}")