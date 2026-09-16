import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RQ1_PATH = PROJECT_ROOT / "results" / "rq1_baseline" / "rq1_summary.csv"
RQ2_PATH = PROJECT_ROOT / "results" / "rq2_depth" / "rq2_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "rq2_depth" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_comparison_plot():
    # Load data
    try:
        df1 = pd.read_csv(RQ1_PATH)
        df1['Architecture'] = 'Shallow (1-Layer)'
        
        df2 = pd.read_csv(RQ2_PATH)
        df2['Architecture'] = 'Deep (2-Layer)'
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure both RQ1 and RQ2 have been run.")
        return

    # Combine datasets
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Format rule names for plotting
    rule_mapping = {
        'gradient_descent': 'Gradient Descent',
        'chl': 'CHL',
        'hebbian': 'Pure Hebbian',
        'anti_hebbian': 'Anti-Hebbian',
        'qpc': 'QPC'
    }
    df_combined['learning_rule'] = df_combined['learning_rule'].replace(rule_mapping)

    # Set up the plot style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # Create grouped bar chart
    ax = sns.barplot(
        data=df_combined,
        x='learning_rule',
        y='relative_improvement_pct',
        hue='Architecture',
        palette=['#4C72B0', '#DD8452'], # Distinct, professional colors
        capsize=.1,
        err_kws={'linewidth': 1.5}
    )

    # Formatting
    plt.title('Impact of Network Depth on Speed of Acquisition (Go-CLS)', fontsize=14, pad=15)
    plt.xlabel('Learning Rule', fontsize=12, labelpad=10)
    plt.ylabel('Relative Validation Improvement (%)', fontsize=12, labelpad=10)
    plt.legend(title='Cortical Architecture', loc='upper right')
    
    # Despine for cleaner look
    sns.despine(left=True, bottom=True)
    
    # Save the figure
    output_path = OUTPUT_DIR / "rq1_vs_rq2_improvement.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved to {output_path}")

if __name__ == "__main__":
    generate_comparison_plot()