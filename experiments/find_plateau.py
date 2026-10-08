
# experiments/find_plateau.py
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from experiments.config import ExperimentConfig
from experiments.runner import run_experiment

config = replace(
    ExperimentConfig(seed=42),
    max_epochs=2000,
    patience=10**6,       # disable early stopping
)

result = run_experiment("gradient_descent", config)
losses = result.validation_losses

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation loss")
ax.set_title("GD training to 2000 epochs — where is the plateau?")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PROJECT_ROOT / "results" / "plateau_analysis.png", dpi=200)
plt.show()

# Print last 20 epoch changes
print("Last 20 epochs, per-epoch change:")
for i in range(-20, 0):
    print(f"  epoch {len(losses)+i}: loss={losses[i]:.6f}, "
          f"delta={losses[i]-losses[i-1]:+.8f}")