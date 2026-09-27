import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure paths are correct
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_cao_heatmap():
    # 1. Define the Cao 2020 Coordinate Space based on your results
    # gamma (x-axis) = Top-Down Feedback (-1, 0, 1)
    # eta (y-axis) = Hebbian Plasticity (-0.1, 0, 0.1)
    
    data = [
        # Rule, gamma, eta, RQ1_Improvement, RQ2_Delta
        ('QPC', -1, 0, 4.38, 6.21),
        ('Gradient Descent', 0, 0, 15.40, 8.32),
        ('CHL', 1, 0, 20.98, 3.22),
        ('Pure Hebbian', 0, 0.1, 9.81, 6.52),
        ('Anti-Hebbian', 0, -0.1, 9.76, 6.50)
    ]
    
    df = pd.DataFrame(data, columns=['Rule', 'gamma', 'eta', 'RQ1_Improvement', 'RQ2_Delta'])
    
    # 2. Create the Grid / Pivot Tables
    # We want eta on the y-axis (descending) and gamma on the x-axis
    grid_rq1 = df.pivot(index='eta', columns='gamma', values='RQ1_Improvement')
    grid_rq1 = grid_rq1.sort_index(ascending=False) # 0.1 at top, -0.1 at bottom
    
    grid_delta = df.pivot(index='eta', columns='gamma', values='RQ2_Delta')
    grid_delta = grid_delta.sort_index(ascending=False)

    # 3. Setup the Visual Landscape (1 row, 2 columns)
    sns.set_theme(style="white", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Common heatmap settings
    annot_kws = {"size": 14, "weight": "bold"}
    
    # --- Subplot 1: RQ1 Improvement (Shallow Cortex) ---
    sns.heatmap(
        grid_rq1, ax=axes[0], annot=True, fmt=".2f", cmap="YlGnBu", 
        cbar_kws={'label': 'Relative Improvement (%)'}, annot_kws=annot_kws,
        linewidths=1, linecolor='black', vmin=0, vmax=25
    )
    axes[0].set_title('RQ1: Shallow Network Acquisition\nAcross the Cao (2020) Space', pad=15, fontweight='bold')
    axes[0].set_xlabel(r'Top-Down Feedback ($\gamma$)', fontsize=12, labelpad=10)
    axes[0].set_ylabel(r'Hebbian Plasticity ($\eta$)', fontsize=12, labelpad=10)

    # --- Subplot 2: RQ2 Delta (Impact of Depth) ---
    sns.heatmap(
        grid_delta, ax=axes[1], annot=True, fmt=".2f", cmap="OrRd", 
        cbar_kws={'label': 'Delta Improvement Jump (%)'}, annot_kws=annot_kws,
        linewidths=1, linecolor='black', vmin=0, vmax=10
    )
    axes[1].set_title('RQ2: Dependency on Network Depth\nAcross the Cao (2020) Space', pad=15, fontweight='bold')
    axes[1].set_xlabel(r'Top-Down Feedback ($\gamma$)', fontsize=12, labelpad=10)
    axes[1].set_ylabel(r'Hebbian Plasticity ($\eta$)', fontsize=12, labelpad=10)

    # Save and show
    plt.tight_layout()
    output_path = OUTPUT_DIR / "cao_parameter_space_heatmaps.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmaps successfully generated and saved to: {output_path}")

if __name__ == "__main__":
    generate_cao_heatmap()