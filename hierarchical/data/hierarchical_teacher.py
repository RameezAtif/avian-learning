"""
Hierarchical synthetic data with TRUE nested structure.

Samples belong to one of K sub-categories. Sub-categories are grouped
into S super-categories. Labels encode BOTH:
  - which super-category the sample is in (shared across sub-cats)
  - which sub-category within the super (unique to each sub-cat)

This creates non-trivial off-diagonal structure in yy^T, which is
what the CHL correlation term needs to do structural work.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class HierarchicalDataset:
    x: np.ndarray                 # (N, input_dim)
    y: np.ndarray                 # (N, n_super + n_sub) — structured labels
    super_means: np.ndarray       # (n_super, input_dim)
    sub_means: np.ndarray         # (n_sub, input_dim)


def generate_hierarchical_data(
    n_super: int = 2,             # number of super-categories
    n_sub_per_super: int = 10,    # sub-categories per super
    samples_per_sub: int = 15,
    input_dim: int = 50,
    within_category_noise: float = 1.5,
    super_separation: float = 3.0,
    seed: int | None = None,
) -> HierarchicalDataset:
    """
    Generate hierarchical data with two levels.

    Super-category means are far apart (super_separation).
    Sub-category means are super_mean + small offset, so sub-cats
    within a super are similar but not identical.

    Labels: [super_one_hot | sub_one_hot]
      - super_one_hot has 1 at the sample's super index
      - sub_one_hot has 1 at the sample's global sub index
    """
    rng = np.random.default_rng(seed)

    n_sub = n_super * n_sub_per_super

    # Super-category means: far apart
    super_means = rng.normal(0.0, super_separation, (n_super, input_dim))

    # Sub-category means: super mean + small offset
    sub_means = np.zeros((n_sub, input_dim))
    for s in range(n_super):
        for k in range(n_sub_per_super):
            global_idx = s * n_sub_per_super + k
            sub_means[global_idx] = (
                super_means[s] + rng.normal(0.0, 0.5, input_dim)
            )

    X_list = []
    Y_list = []

    for s in range(n_super):
        for k in range(n_sub_per_super):
            global_idx = s * n_sub_per_super + k

            x_samples = sub_means[global_idx] + rng.normal(
                0.0, within_category_noise,
                (samples_per_sub, input_dim)
            )

            # Structured label: super one-hot concatenated with sub one-hot
            y_samples = np.zeros((samples_per_sub, n_super + n_sub))
            y_samples[:, s] = 1.0                              # super bit
            y_samples[:, n_super + global_idx] = 1.0           # sub bit

            X_list.append(x_samples)
            Y_list.append(y_samples)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    perm = rng.permutation(len(X))
    return HierarchicalDataset(
        x=X[perm], y=Y[perm],
        super_means=super_means, sub_means=sub_means,
    )