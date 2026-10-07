from dataclasses import dataclass

import numpy as np


@dataclass
class NotebookMemory:
    """
    Stores one encoded experience in the Notebook.

    Attributes
    ----------
    pattern : np.ndarray
        Sparse binary Hopfield pattern with shape (notebook_dim,).

    x : np.ndarray
        Input associated with this memory.

    y : np.ndarray
        Target output associated with this memory.

    snr : float
        Signal-to-noise ratio of the experience. Used to weight
        replay probability under the Go-CLS framework (Sun et al. 2023):
        high-SNR (predictable) experiences are replayed more often.
    """

    pattern: np.ndarray
    x: np.ndarray
    y: np.ndarray
    snr: float = 1.0


@dataclass
class ReplayResult:
    """
    Stores the result of a Notebook retrieval.

    Attributes
    ----------
    pattern : np.ndarray
        Retrieved sparse binary Notebook state.

    x : np.ndarray
        Reconstructed/retrieved input.

    y : np.ndarray
        Reconstructed/retrieved target.

    memory_index : int
        Index of the stored memory that was retrieved.

    similarity : float
        Similarity between the retrieved pattern and the
        closest stored pattern.
    """

    pattern: np.ndarray
    x: np.ndarray
    y: np.ndarray
    memory_index: int
    similarity: float


class SparseHopfieldNotebook:
    """
    Sparse Hopfield Notebook for fast episodic encoding and replay.

    The Notebook stores sparse binary patterns and associates
    each pattern with a teacher-generated (x, y) experience.

    The implementation contains:

        1. Sparse binary memory generation
        2. Hebbian recurrent encoding
        3. Pattern retrieval through recurrent dynamics
        4. Association between Notebook memories and experiences

    In the main Go-CLS experiment, the notebook is an episodic store whose
    contents are reactivated (``replay_stored_batch``). Replay sampling is
    weighted by the SNR of each stored experience, implementing the
    predictability-gated consolidation mechanism from Sun et al. (2023):
    high-SNR experiences (e.g. pristine tutor samples) are replayed more
    often than low-SNR ones (e.g. noisy practice samples). If no SNR
    information is provided, replay is uniform.

    Parameters
    ----------
    notebook_dim : int
        Number of units in the Notebook.

    sparsity : float
        Fraction of Notebook units active in each memory.

    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        notebook_dim: int,
        sparsity: float,
        seed: int | None = None,
    ):
        if notebook_dim <= 0:
            raise ValueError(
                "notebook_dim must be greater than 0."
            )

        if not 0 < sparsity <= 1:
            raise ValueError(
                "sparsity must be in the interval (0, 1]."
            )

        active_units = int(
            round(notebook_dim * sparsity)
        )

        if active_units <= 0:
            raise ValueError(
                "sparsity produces zero active units."
            )

        self.notebook_dim = notebook_dim
        self.sparsity = sparsity
        self.active_units = active_units

        self.rng = np.random.default_rng(seed)

        # Sparse Hopfield recurrent weights.
        self.recurrent_weights = np.zeros(
            (notebook_dim, notebook_dim),
            dtype=float,
        )

        # Stored memories.
        self.memories: list[NotebookMemory] = []

    def _create_sparse_pattern(self) -> np.ndarray:
        """
        Create a sparse binary Notebook pattern.
        Exactly `active_units` units are set to 1.
        """

        pattern = np.zeros(
            self.notebook_dim,
            dtype=float,
        )

        active_indices = self.rng.choice(
            self.notebook_dim,
            size=self.active_units,
            replace=False,
        )

        pattern[active_indices] = 1.0

        return pattern

    def encode(
        self,
        x: np.ndarray,
        y: np.ndarray,
        snr: float = 1.0,
    ) -> NotebookMemory:
        """
        One-shot encode an experience.

        Parameters
        ----------
        x : np.ndarray
            Input experience.

        y : np.ndarray
            Target experience.

        snr : float
            Signal-to-noise ratio of the experience. Defaults to 1.0.
            Under the Go-CLS framework this determines replay probability.

        Returns
        -------
        NotebookMemory
            Newly stored memory.
        """

        if x.ndim != 1:
            raise ValueError(
                "x must be a 1D array."
            )

        if y.ndim != 0:
            raise ValueError(
                "y must be a scalar array."
            )

        pattern = self._create_sparse_pattern()

        # Hebbian Hopfield storage:
        #   W <- W + ξ ξ^T
        # Remove diagonal so a neuron does not reinforce itself.
        self.recurrent_weights += np.outer(
            pattern,
            pattern,
        )

        np.fill_diagonal(
            self.recurrent_weights,
            0.0,
        )

        memory = NotebookMemory(
            pattern=pattern.copy(),
            x=x.copy(),
            y=y.copy(),
            snr=float(snr),
        )

        self.memories.append(memory)

        return memory

    def encode_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        snr_values: np.ndarray | None = None,
    ) -> None:
        """
        Encode a batch of teacher experiences.

        Parameters
        ----------
        x : np.ndarray
            Inputs with shape (N, input_dim).

        y : np.ndarray
            Targets with shape (N,).

        snr_values : np.ndarray or None
            Per-sample SNR values with shape (N,). If None, all
            experiences are assigned snr = 1.0 and replay is uniform.
        """

        if x.ndim != 2:
            raise ValueError(
                "x must be a 2D array."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1D array."
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "x and y must contain the same number "
                "of examples."
            )

        if snr_values is not None:
            snr_values = np.asarray(snr_values, dtype=float)
            if snr_values.shape != (x.shape[0],):
                raise ValueError(
                    "snr_values must have shape (N,)."
                )

        for index in range(x.shape[0]):
            snr = 1.0 if snr_values is None else float(snr_values[index])
            self.encode(
                x=x[index],
                y=np.asarray(y[index]),
                snr=snr,
            )

    def _activate(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Perform one synchronous sparse Hopfield update.
        """

        activation = (
            self.recurrent_weights @ state
        )

        active_indices = np.argpartition(
            activation,
            -self.active_units,
        )[-self.active_units:]

        new_state = np.zeros(
            self.notebook_dim,
            dtype=float,
        )

        new_state[active_indices] = 1.0

        return new_state

    def retrieve(
        self,
        cycles: int = 9,
        initial_state: np.ndarray | None = None,
    ) -> ReplayResult:
        """
        Retrieve a stored memory through Hopfield dynamics.
        Used as a diagnostic only.
        """

        if len(self.memories) == 0:
            raise RuntimeError(
                "Cannot retrieve from an empty Notebook."
            )

        if cycles <= 0:
            raise ValueError(
                "cycles must be greater than 0."
            )

        if initial_state is None:
            state = self._create_sparse_pattern()

        else:
            state = np.asarray(
                initial_state,
                dtype=float,
            ).copy()

            if state.shape != (
                self.notebook_dim,
            ):
                raise ValueError(
                    "initial_state has incorrect shape."
                )

            state = self._make_sparse(state)

        for _ in range(cycles):
            state = self._activate(state)

        similarities = np.array(
            [
                self._similarity(
                    state,
                    memory.pattern,
                )
                for memory in self.memories
            ]
        )

        memory_index = int(
            np.argmax(similarities)
        )

        similarity = float(
            similarities[memory_index]
        )

        memory = self.memories[
            memory_index
        ]

        return ReplayResult(
            pattern=state.copy(),
            x=memory.x.copy(),
            y=memory.y.copy(),
            memory_index=memory_index,
            similarity=similarity,
        )

    def replay_batch(
        self,
        num_replays: int,
        cycles: int = 9,
    ) -> list[ReplayResult]:
        """
        Retrieve several memories using Hopfield dynamics.
        Diagnostic only.
        """

        if num_replays <= 0:
            raise ValueError(
                "num_replays must be greater than 0."
            )

        results = []

        for _ in range(num_replays):
            results.append(
                self.retrieve(
                    cycles=cycles,
                )
            )

        return results

    def replay_stored_batch(self, num_replays: int) -> list[ReplayResult]:
        """
        Reactivate stored experiences with SNR-gated sampling.

        Sampling probability is proportional to the SNR of each stored
        experience. This implements the Go-CLS consolidation rule from
        Sun et al. (2023): high-predictability (high-SNR) experiences
        are replayed more frequently than low-SNR noise.

        If all SNRs are equal (or no SNR was provided), this reduces to
        uniform sampling.
        """
        if num_replays <= 0:
            raise ValueError("num_replays must be greater than 0.")
        if not self.memories:
            raise RuntimeError("Cannot retrieve from an empty Notebook.")

        # Build SNR weight vector
        snr_weights = np.array(
            [m.snr for m in self.memories], dtype=float,
        )

        # Guard against non-positive or zero-sum weights
        snr_weights = np.clip(snr_weights, a_min=1e-12, a_max=None)
        total = snr_weights.sum()

        if total <= 0 or not np.isfinite(total):
            # Fall back to uniform sampling
            probabilities = None
        else:
            probabilities = snr_weights / total

        # Sample indices with SNR weighting
        if probabilities is None:
            indices = self.rng.integers(
                0, len(self.memories), size=num_replays,
            )
        else:
            indices = self.rng.choice(
                len(self.memories),
                size=num_replays,
                p=probabilities,
                replace=True,
            )

        return [
            ReplayResult(
                pattern=self.memories[int(i)].pattern.copy(),
                x=self.memories[int(i)].x.copy(),
                y=self.memories[int(i)].y.copy(),
                memory_index=int(i),
                similarity=1.0,
            )
            for i in indices
        ]

    def __len__(self) -> int:
        return len(self.memories)

    def _make_sparse(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Convert an arbitrary state into a sparse binary state.
        """

        active_indices = np.argpartition(
            state,
            -self.active_units,
        )[-self.active_units:]

        sparse_state = np.zeros(
            self.notebook_dim,
            dtype=float,
        )

        sparse_state[active_indices] = 1.0

        return sparse_state

    @staticmethod
    def _similarity(
        pattern_a: np.ndarray,
        pattern_b: np.ndarray,
    ) -> float:
        """
        Normalized binary overlap between two patterns.
        """

        return float(
            np.sum(pattern_a * pattern_b)
            / np.sum(pattern_b)
        )