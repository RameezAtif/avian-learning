"""Quick test: does a tight bottleneck produce progressive differentiation?"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from rq3.run_rq3 import train_one

BOTTLENECKS = [2, 3, 4, 6, 8, 16, 32]

fig, axes = plt.subplots(1, len(BOTTLENECKS), figsize=(20, 4), sharey=True)

for ax, h_dim in zip(axes, BOTTLENECKS):
    result = train_one("gradient_descent", seed=42, hidden_dim=h_dim)
    history = result["history"]

    sv_cols = [f"sv_{i}" for i in range(min(8, h_dim))]
    for col in sv_cols:
        values = [h[col] for h in history]
        ax.plot(values, alpha=0.7)

    ax.set_title(f"hidden_dim={h_dim}")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Singular value")
fig.suptitle("Progressive differentiation vs bottleneck size (GD, seed 42)")
fig.tight_layout()
fig.savefig(PROJECT_ROOT / "rq3" / "results" / "bottleneck_sweep.png", dpi=200)
plt.show()