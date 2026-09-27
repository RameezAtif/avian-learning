"""
Diagnostic: track the two terms of the CHL W2 update separately.

If term_B (the label-correlation contrast) is near zero on our
Gaussian data, then CHL collapses to GD. That is the supervisor's
hypothesis and this script tests it directly.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.teacher import create_teacher, generate_teacher_experiences
from experiments.config import ExperimentConfig
from learning.chl import ContrastiveHebbianRule
from models.student import Student
from memory.notebook import SparseHopfieldNotebook


def main():
    config = ExperimentConfig(seed=42)
    B = config.replays_per_epoch

    # Data
    teacher = create_teacher(config.input_dim, seed=42)
    tutor = generate_teacher_experiences(
        teacher, config.num_tutor_examples,
        1.0 / config.tutor_snr, 43, "tutor",
    )
    practice = generate_teacher_experiences(
        teacher, config.num_practice_examples,
        1.0 / config.practice_snr, 44, "practice",
    )
    train_x = np.vstack((tutor.x, practice.x))
    train_y = np.concatenate((tutor.y, practice.y))

    # Student + notebook
    student = Student(input_dim=config.input_dim,
                      hidden_dim=config.hidden_dim, seed=42)
    notebook = SparseHopfieldNotebook(
        notebook_dim=config.notebook_dim,
        sparsity=config.notebook_sparsity, seed=42,
    )
    notebook.encode_batch(x=train_x, y=train_y)

    # CHL rule with diagnostics enabled
    rule = ContrastiveHebbianRule(
        learning_rate=config.learning_rate,
        gamma=1.0,
        eta=0.0,
        gradient_clip=None,   # disable clipping so we see raw magnitudes
        update_w2=True,
    )

    rows = []
    for epoch in range(1, 201):
        results = notebook.replay_stored_batch(B)
        x = np.stack([r.x for r in results])
        y = np.asarray([r.y for r in results])

        out = student.forward(x)
        rule.calculate_update(
            x=x, y=y, h=out.h_ff, y_hat=out.y_hat,
            w1=student.W1, w2=student.W2,
        )

        terms = rule.last_terms
        rows.append({
            "epoch": epoch,
            "norm_termA_error": np.linalg.norm(terms["delta_w2_termA_error"]),
            "norm_termB_correlation": np.linalg.norm(terms["delta_w2_termB_correlation"]),
            "corr_coefficient": terms["corr_coefficient"],
            "norm_hebb": np.linalg.norm(terms["delta_w1_hebb"]),
            "norm_w1_update": np.linalg.norm(terms["delta_w1_chl"]),
        })

        # Apply the update manually (mimicking the replay loop)
        upd = rule.calculate_update(
            x=x, y=y, h=out.h_ff, y_hat=out.y_hat,
            w1=student.W1, w2=student.W2,
        )
        student.W1 += upd.delta_w1
        student.W2 += upd.delta_w2

    df = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "chl_term_diagnostic.csv", index=False)

    print(df.describe())
    print(f"\nSaved: {out_dir / 'chl_term_diagnostic.csv'}")


if __name__ == "__main__":
    main()