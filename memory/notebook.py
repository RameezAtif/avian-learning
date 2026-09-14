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
    """

    pattern: np.ndarray
    x: np.ndarray
    y: np.ndarray


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
    contents are faithfully reactivated (``replay_stored_batch``).  The
    recurrent Hopfield dynamics remain a retrieval-quality diagnostic only:
    random-cue retrieval can preferentially return a subset of stored
    associations, so it is not used to choose training experiences unless
    that retrieval process is itself the subject of an explicitly separate
    experiment.

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
        #
        # W_notebook[i, j] describes the influence of
        # Notebook unit j on Notebook unit i.
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
    ) -> NotebookMemory:
        """
        One-shot encode an experience.

        A new sparse binary pattern is created and associated
        with the input-target pair.

        Hebbian outer-product learning is then used to make
        the sparse pattern an attractor of the recurrent
        Hopfield network.

        Parameters
        ----------
        x : np.ndarray
            Input experience.

        y : np.ndarray
            Target experience.

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
        #
        # W <- W + ξ ξ^T
        #
        # We remove the diagonal so that a neuron does not
        # directly reinforce itself.
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
        )

        self.memories.append(memory)

        return memory

    def encode_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Encode a batch of teacher experiences.
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

        for index in range(x.shape[0]):
            self.encode(
                x=x[index],
                y=np.asarray(y[index]),
            )

    def _activate(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Perform one synchronous sparse Hopfield update.

        The largest `active_units` activations become active.
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

        Parameters
        ----------
        cycles : int
            Number of recurrent update cycles.

        initial_state : np.ndarray or None
            Initial Notebook activity. If None, a random sparse
            pattern is created.

        Returns
        -------
        ReplayResult
            Retrieved memory and associated experience.
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

        # Synchronous recurrent updates.
        for _ in range(cycles):
            state = self._activate(state)

        # Find the closest stored memory.
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
        Retrieve several memories for one replay epoch.

        Memories are retrieved independently using random
        sparse initial Notebook activity.
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
        """Uniformly reactivate stored experiences with faithful recall.

        This operationalizes the Go-CLS assumption of accurate notebook
        recall. Sampling is uniform over stored episodes; it deliberately
        avoids the retrieval bias observed with random-cue Hopfield dynamics.
        The dynamics method remains available as a diagnostic, but is not an
        unverified source of training samples.
        """
        if num_replays <= 0:
            raise ValueError("num_replays must be greater than 0.")
        if not self.memories:
            raise RuntimeError("Cannot retrieve from an empty Notebook.")
        indices = self.rng.integers(0, len(self.memories), size=num_replays)
        return [ReplayResult(
            pattern=self.memories[int(i)].pattern.copy(), x=self.memories[int(i)].x.copy(),
            y=self.memories[int(i)].y.copy(), memory_index=int(i), similarity=1.0,
        ) for i in indices]

    def __len__(self) -> int:
        """
        Return the number of stored memories.
        """

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
        Calculate normalized binary overlap between two
        Notebook patterns.
        """

        return float(
            np.sum(pattern_a * pattern_b)
            / np.sum(pattern_b)
        )
