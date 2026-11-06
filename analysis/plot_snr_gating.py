"""
Generates the SNR-gating comparison bar chart for Section 4.5.
Run: python analysis/plot_snr_gating.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "baseline_go_cls_snr_lownoise" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Exact values from the LOW-NOISE REGIME run
rules = ["Gradient Descent", "CHL", "QPC", "Pure Hebbian", "Anti-Hebbian"]
uniform_mean = [0.253, 0.756, 0.471, 0.277, 0.299]
uniform_std  = [0.040, 0.224, 0.093, 0.039, 0.040]
snr_mean     = [0.223, 0.463, 0.471, 0.279, 0.303]
snr_std      = [0.038, 0.101, 0.093, 0.040, 0.041]

x = np.arange(len(rules))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - width/2, uniform_mean, width, yerr=uniform_std, capsize=4,
       label="Uniform sampling", color="#A0CBE8", edgecolor="black", linewidth=0.7)
ax.bar(x + width/2, snr_mean, width, yerr=snr_std, capsize=4,
       label="SNR-gated sampling", color="#4E79A7", edgecolor="black", linewidth=0.7)

# Annotate the CHL improvement explicitly
chl_idx = rules.index("CHL")
ax.annotate(
    f"↓ {uniform_mean[chl_idx] - snr_mean[chl_idx]:.2f}",
    xy=(x[chl_idx], snr_mean[chl_idx]),
    xytext=(0, -45),
    textcoords="offset points",
    ha="center",
    fontsize=11,
    fontweight="bold",
    color="darkred",
)

ax.set_xticks(x)
ax.set_xticklabels(rules, rotation=15, ha="right")
ax.set_ylabel("Final validation loss (lower is better)")
ax.set_title("SNR-Gated vs Uniform Replay (Low-Noise Regime: Tutor SNR=5, Practice SNR=1)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out_path = OUT / "snr_gating_results.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")