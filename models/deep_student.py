from dataclasses import dataclass
import numpy as np

@dataclass
class DeepStudentOutput:
    """
    Stores the outputs produced by the deep student network.
    h_ff remains named h_ff to maintain structural naming conventions.
    """
    h_ff: np.ndarray       # Cortex state (h1)
    h_readout: np.ndarray  # Deep readout state (h2)
    y_hat: np.ndarray      # Final prediction


class DeepStudent:
    """
    Linear three-weight-matrix student network (RQ2 Deep Cortex).
    
    The forward pass is:
        h_cortex = W1 x           (Biological Plasticity)
        h_readout = W2 h_cortex   (Supervised GD Readout)
        y_hat = W3 h_readout      (Supervised GD Output)
    """
    def __init__(self, input_dim: int, cortex_dim: int, readout_dim: int, seed: int | None = None):
        if input_dim <= 0 or cortex_dim <= 0 or readout_dim <= 0:
            raise ValueError("Dimensions must be greater than 0.")
        
        rng = np.random.default_rng(seed)
        
        # W1: Brainstem to Cortex (Plastic layer)
        self.W1 = rng.normal(loc=0.0, scale=1.0 / np.sqrt(input_dim), size=(cortex_dim, input_dim))
        
        # W2: Cortex to Readout (GD layer)
        self.W2 = rng.normal(loc=0.0, scale=1.0 / np.sqrt(cortex_dim), size=(readout_dim, cortex_dim))
        
        # W3: Readout to Final Target (GD layer)
        self.W3 = rng.normal(loc=0.0, scale=1.0 / np.sqrt(readout_dim), size=(1, readout_dim))

    def get_effective_w2(self) -> np.ndarray:
        """
        Collapses the two downstream readout matrices into a single matrix.
        Shape: (1, readout_dim) @ (readout_dim, cortex_dim) = (1, cortex_dim).
        This allows the biological Master Equation to receive the correct top-down error
        without requiring a rewrite of the plasticity rules.
        """
        return self.W3 @ self.W2

    def forward(self, x: np.ndarray) -> DeepStudentOutput:
        if x.ndim == 1:
            h_cortex = self.W1 @ x
            h_readout = self.W2 @ h_cortex
            y_hat = (self.W3 @ h_readout).squeeze()
            return DeepStudentOutput(h_ff=h_cortex, h_readout=h_readout, y_hat=y_hat)
        
        if x.ndim == 2:
            h_cortex = x @ self.W1.T
            h_readout = h_cortex @ self.W2.T
            y_hat = (h_readout @ self.W3.T).squeeze(axis=1)
            return DeepStudentOutput(h_ff=h_cortex, h_readout=h_readout, y_hat=y_hat)

        raise ValueError("x must have shape (input_dim,) or (num_examples, input_dim).")