import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore") # Suppress seaborn layout warnings

def generate_visuals():
    # 1. Setup Publication Aesthetics
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': '--'})
    palette = sns.color_palette("colorblind", 5)

    # Paths based on your repository structure
    RESULTS_DIR = "results/rq1_baseline"
    HISTORIES_PATH = os.path.join(RESULTS_DIR, "rq1_histories.csv")
    SUMMARY_PATH = os.path.join(RESULTS_DIR, "rq1_summary.csv")
    
    OUT_DIR = os.path.join(RESULTS_DIR, "publication_figures")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    hist_df = pd.read_csv(HISTORIES_PATH)
    sum_df = pd.read_csv(SUMMARY_PATH)

    # Safely find the exact column names your framework used
    best_val_col = [col for col in sum_df.columns if 'best' in col.lower() and ('val' in col.lower() or 'loss' in col.lower())][0]
    stab_var_col = [col for col in sum_df.columns if 'stabil' in col.lower()][0]

    # Format the rule names for the legend
    rule_names = {
        'gradient_descent': 'Gradient Descent',
        'chl': 'CHL (Surrogate)',
        'qpc': 'Quasi-Predictive Coding',
        'hebbian': 'Pure Hebbian',
        'anti_hebbian': 'Anti-Hebbian'
    }
    hist_df['Rule'] = hist_df['learning_rule'].map(rule_names)
    sum_df['Rule'] = sum_df['learning_rule'].map(rule_names)

    # =====================================================================
    # FIGURE 1: Shaded Validation Learning Curves
    # =====================================================================
    print("Generating Figure 1: Validation Learning Curves...")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=hist_df, 
        x='epoch', 
        y='validation_loss', 
        hue='Rule', 
        palette=palette,
        linewidth=2.5
    )
    
    plt.title("Generalisation Trajectory Across 5 Random Seeds (SNR = 1)", pad=15, fontweight='bold')
    plt.xlabel("Epoch (Go-CLS Replay Cycles)", fontweight='bold')
    plt.ylabel("Validation Loss (Mean Squared Error)", fontweight='bold')
    plt.legend(title="Learning Rule", frameon=True, shadow=True)
    sns.despine() 
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig1_Validation_Trajectory.png"), dpi=300)
    plt.close()

    # =====================================================================
    # FIGURE 2: Final Performance Boxplots
    # =====================================================================
    print("Generating Figure 2: Final Performance Distributions...")
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        data=sum_df, 
        x='Rule', 
        y=best_val_col,
        hue='Rule', 
        palette=palette,
        width=0.6,
        boxprops=dict(alpha=0.8),
        legend=False
    )
    sns.stripplot(data=sum_df, x='Rule', y=best_val_col, color='black', alpha=0.6, jitter=True)
    
    plt.title("Distribution of Best Validation Loss Achieved", pad=15, fontweight='bold')
    plt.xlabel("") 
    plt.ylabel("Best Validation Loss", fontweight='bold')
    plt.xticks(rotation=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig2_Performance_Boxplot.png"), dpi=300)
    plt.close()

    # =====================================================================
    # TABLE 1: Publication Summary Table (Mean ± SD)
    # =====================================================================
    print("Generating Publication Table...")
    
    stats = sum_df.groupby('Rule').agg({
        best_val_col: ['mean', 'std'],
        stab_var_col: ['mean', 'std']
    })
    
    pub_table = pd.DataFrame()
    pub_table['Final Validation Loss'] = stats[best_val_col].apply(
        lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
    )
    pub_table['Stability Variance'] = stats[stab_var_col].apply(
        lambda x: f"{x['mean']:.6f} ± {x['std']:.6f}", axis=1
    )
    
    pub_table.to_csv(os.path.join(OUT_DIR, "Table1_Publication_Metrics.csv"))
    print("\n--- PUBLICATION METRICS TABLE ---")
    print(pub_table.to_markdown())
    print("---------------------------------")
    print(f"\nAll high-res visuals saved to: {OUT_DIR}")

    sns.set_context("paper", font_scale=1.3)
    sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': '--'})
    palette = sns.color_palette("colorblind", 5)

    RESULTS_DIR = "results/rq1_baseline"
    HISTORIES_PATH = os.path.join(RESULTS_DIR, "rq1_histories.csv")
    SUMMARY_PATH = os.path.join(RESULTS_DIR, "rq1_summary.csv")
    OUT_DIR = os.path.join(RESULTS_DIR, "publication_figures")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    hist_df = pd.read_csv(HISTORIES_PATH)
    sum_df = pd.read_csv(SUMMARY_PATH)

    # Dynamically find column names to prevent Pandas errors
    best_val_col = [col for col in sum_df.columns if 'best' in col.lower() and ('val' in col.lower() or 'loss' in col.lower())][0]
    stab_var_col = [col for col in sum_df.columns if 'stabil' in col.lower()][0]
    epochs_col = [col for col in sum_df.columns if 'epoch' in col.lower() and 'exec' in col.lower()][0]

    rule_names = {
        'gradient_descent': 'Gradient Descent',
        'chl': 'CHL (Surrogate)',
        'qpc': 'Quasi-Predictive Coding',
        'hebbian': 'Pure Hebbian',
        'anti_hebbian': 'Anti-Hebbian'
    }
    hist_df['Rule'] = hist_df['learning_rule'].map(rule_names)
    sum_df['Rule'] = sum_df['learning_rule'].map(rule_names)

    # =====================================================================
    # FIGURE 3: Replay Loss Trajectory (Sleep Consolidation)
    # =====================================================================
    print("Generating Figure 3: Replay Loss Curves...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hist_df, x='epoch', y='replay_loss', hue='Rule', palette=palette, linewidth=2)
    plt.title("Sleep Replay Loss Trajectory (Consolidation Efficiency)", pad=15, fontweight='bold')
    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel("Replay Loss", fontweight='bold')
    plt.legend(title="Learning Rule")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig3_Replay_Loss.png"), dpi=300)
    plt.close()

    # =====================================================================
    # FIGURE 4: Epochs to Convergence (Speed of Acquisition)
    # =====================================================================
    print("Generating Figure 4: Epochs Executed...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sum_df, x='Rule', y=epochs_col, palette=palette, capsize=.1, errwidth=2)
    plt.title("Speed of Acquisition (Epochs until Go-CLS Early Stopping)", pad=15, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Epochs Executed", fontweight='bold')
    plt.xticks(rotation=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig4_Epochs_Executed.png"), dpi=300)
    plt.close()

    # =====================================================================
    # FIGURE 5: Representational Stability 
    # =====================================================================
    print("Generating Figure 5: Stability Variance...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=sum_df, x='Rule', y=stab_var_col, palette=palette, width=0.5)
    sns.stripplot(data=sum_df, x='Rule', y=stab_var_col, color='black', alpha=0.6, jitter=True)
    plt.title("Representational Stability at Convergence (Lower is more stable)", pad=15, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Stability Variance", fontweight='bold')
    plt.xticks(rotation=15)
    
    # Use a log scale if variance numbers are microscopic (e.g., 0.0001)
    if sum_df[stab_var_col].mean() < 0.01:
        plt.yscale('log')
        plt.ylabel("Stability Variance (Log Scale)", fontweight='bold')
        
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig5_Stability_Variance.png"), dpi=300)
    plt.close()

    print(f"Success! Advanced visuals saved to {OUT_DIR}")


    sns.set_context("paper", font_scale=1.3)
    sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': '--'})
    palette = sns.color_palette("colorblind", 5)

    RESULTS_DIR = "results/rq1_baseline"
    HISTORIES_PATH = os.path.join(RESULTS_DIR, "rq1_histories.csv")
    OUT_DIR = os.path.join(RESULTS_DIR, "publication_figures")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    hist_df = pd.read_csv(HISTORIES_PATH)

    rule_names = {
        'gradient_descent': 'Gradient Descent',
        'chl': 'CHL (Surrogate)',
        'qpc': 'Quasi-Predictive Coding',
        'hebbian': 'Pure Hebbian',
        'anti_hebbian': 'Anti-Hebbian'
    }
    hist_df['Rule'] = hist_df['learning_rule'].map(rule_names)

    # =====================================================================
    # NORMALIZE THE DATA
    # Divides every loss value by the starting loss of its specific seed
    # Forces all curves to start exactly at 1.0 (100% initial error)
    # =====================================================================
    print("Calculating normalized trajectories...")
    hist_df['normalized_loss'] = hist_df.groupby(['learning_rule', 'seed'])['validation_loss'].transform(lambda x: x / x.iloc[0])

    # =====================================================================
    # FIGURE 1: Normalized Validation Trajectory (Speed of Acquisition)
    # =====================================================================
    print("Generating Normalized Trajectory Plot...")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=hist_df, 
        x='epoch', 
        y='normalized_loss', 
        hue='Rule', 
        style='Rule',      # Uses distinct line styles (solid, dashed, dotted)
        markers=True,      # Adds geometric markers
        dashes=True,
        markevery=30,      # Spaces out markers so it doesn't look cluttered
        palette=palette,
        linewidth=2.5,
        errorbar=None
    )
    
    # Add a horizontal line showing the 50% Go-CLS Acquisition Target
    plt.axhline(y=0.5, color='red', linestyle='--', label='50% Acquisition Threshold', alpha=0.7)
    
    plt.title("Speed of Acquisition: Normalized Trajectory (SNR = 3)", pad=15, fontweight='bold')
    plt.xlabel("Epoch (Go-CLS Replay Cycles)", fontweight='bold')
    plt.ylabel("Normalized Validation Loss (Baseline = 1.0)", fontweight='bold')
    
    # Clean up the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title="Learning Rule", frameon=True, shadow=True)
    
    sns.despine() 
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig1_Normalized_Trajectory.png"), dpi=300)
    plt.close()

    # =====================================================================
    # FIGURE 2: Maximum Relative Improvement % Bar Chart
    # =====================================================================
    print("Generating Relative Improvement Plot...")
    
    # Calculate the percentage drop: (1.0 - lowest_normalized_point) * 100
    improvement_df = hist_df.groupby(['Rule', 'seed'])['normalized_loss'].min().reset_index()
    improvement_df['improvement_pct'] = (1.0 - improvement_df['normalized_loss']) * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=improvement_df, 
        x='Rule', 
        y='improvement_pct', 
        palette=palette,
        capsize=0.1,
        errcolor=".2",
        edgecolor=".2"
    )
    # Overlay the exact seed points
    sns.stripplot(data=improvement_df, x='Rule', y='improvement_pct', color='black', alpha=0.6, jitter=True)
    
    plt.title("Total Knowledge Acquired (Relative Validation Improvement %)", pad=15, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Validation Improvement (%)", fontweight='bold')
    plt.xticks(rotation=15)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig2_Relative_Improvement.png"), dpi=300)
    plt.close()

    print(f"\nAll high-res visuals saved to: {OUT_DIR}")


if __name__ == "__main__":
    generate_visuals()