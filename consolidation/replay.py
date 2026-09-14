from dataclasses import dataclass

import numpy as np

from learning.plasticity import ContinuousPlasticityRule
from memory.notebook import SparseHopfieldNotebook
from models.student import Student


@dataclass
class ReplayMetrics:
    """
    Metrics produced by one replay cycle.

    Attributes
    ----------
    loss : float
        Mean squared error between replayed targets and
        Student predictions.

    replayed_memories : int
        Number of Notebook memories replayed.

    mean_similarity : float
        Mean similarity between retrieved Notebook states
        and their closest stored memories.
    """

    loss: float
    replayed_memories: int
    mean_similarity: float


class SleepReplay:
    """
    Coordinates Notebook replay and Student learning.

    The workflow is:

        Notebook
            ↓
        retrieve memories
            ↓
        replayed x, y
            ↓
        Student forward pass
            ↓
        learning rule
            ↓
        update Student weights
    """

    def __init__(
        self,
        student: Student,
        notebook: SparseHopfieldNotebook,
        learning_rule: ContinuousPlasticityRule,
        replay_cycles: int = 9,
        replays_per_epoch: int = 32,
        update_w2: bool = False,
        replay_mode: str = "stored",
    ):
        if replay_cycles <= 0:
            raise ValueError(
                "replay_cycles must be greater than 0."
            )

        if replays_per_epoch <= 0:
            raise ValueError(
                "replays_per_epoch must be greater than 0."
            )

        self.student = student
        self.notebook = notebook
        self.learning_rule = learning_rule

        self.replay_cycles = replay_cycles
        self.replays_per_epoch = replays_per_epoch
        self.update_w2 = update_w2
        if replay_mode not in {"stored", "hopfield"}:
            raise ValueError("replay_mode must be 'stored' or 'hopfield'.")
        self.replay_mode = replay_mode

    @staticmethod
    def _calculate_loss(
        y: np.ndarray,
        y_hat: np.ndarray,
    ) -> float:
        """
        Calculate mean squared error.

            L = 1/(2N) * sum((y - y_hat)^2)
        """

        return float(
            0.5 * np.mean((y - y_hat) ** 2)
        )

    def run_epoch(self) -> ReplayMetrics:
        """
        Run one sleep-replay epoch.

        Returns
        -------
        ReplayMetrics
            Summary of the replay epoch.
        """

        if len(self.notebook) == 0:
            raise RuntimeError(
                "Cannot run replay with an empty Notebook."
            )

        # -----------------------------------------------------
        # 1. Retrieve memories from Notebook
        # -----------------------------------------------------

        replay_results = (
            self.notebook.replay_stored_batch(self.replays_per_epoch)
            if self.replay_mode == "stored"
            else self.notebook.replay_batch(self.replays_per_epoch, self.replay_cycles)
        )

        # -----------------------------------------------------
        # 2. Convert replayed memories into a batch
        # -----------------------------------------------------

        replayed_x = np.stack(
            [result.x for result in replay_results]
        )

        replayed_y = np.asarray(
            [result.y for result in replay_results],
            dtype=float,
        )

        # -----------------------------------------------------
        # 3. Student forward pass
        # -----------------------------------------------------

        output = self.student.forward(
            replayed_x
        )

        # -----------------------------------------------------
        # 4. Measure loss BEFORE the update
        # -----------------------------------------------------

        loss = self._calculate_loss(
            replayed_y,
            output.y_hat,
        )

        # -----------------------------------------------------
        # 5. Calculate learning-rule update
        # -----------------------------------------------------

        update = self.learning_rule.calculate_update(
            x=replayed_x,
            y=replayed_y,
            h=output.h_ff,
            y_hat=output.y_hat,
            w1=self.student.W1,
            w2=self.student.W2,
        )

        # -----------------------------------------------------
        # 6. Apply W1 update
        # -----------------------------------------------------

        self.student.W1 += update.delta_w1

        # W2 remains configurable.
        if self.update_w2:
            self.student.W2 += update.delta_w2

        # -----------------------------------------------------
        # 7. Calculate mean Notebook retrieval quality
        # -----------------------------------------------------

        similarities = [
            result.similarity
            for result in replay_results
        ]

        mean_similarity = float(
            np.mean(similarities)
        )

        return ReplayMetrics(
            loss=loss,
            replayed_memories=len(replay_results),
            mean_similarity=mean_similarity,
        )

    def run(
        self,
        epochs: int,
    ) -> list[ReplayMetrics]:
        """
        Run multiple sleep-replay epochs.

        Parameters
        ----------
        epochs : int
            Number of replay epochs.

        Returns
        -------
        list[ReplayMetrics]
            One metric object per replay epoch.
        """

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than 0."
            )

        history = []

        for _ in range(epochs):
            metrics = self.run_epoch()
            history.append(metrics)

        return history
