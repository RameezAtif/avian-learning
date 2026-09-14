from dataclasses import dataclass

import numpy as np

from consolidation.replay import ReplayMetrics, SleepReplay


@dataclass
class ConsolidationResult:
    """
    Stores the outcome of a Go-CLS consolidation run.

    Attributes
    ----------
    history : list[ReplayMetrics]
        Replay metrics for every epoch that was executed.

    validation_losses : list[float]
        Validation loss measured after each replay epoch.

    best_validation_loss : float
        Lowest validation loss observed.

    best_epoch : int
        Epoch at which the lowest validation loss occurred.

    epochs_executed : int
        Total number of replay epochs performed.
    """

    history: list[ReplayMetrics]
    validation_losses: list[float]
    best_validation_loss: float
    best_epoch: int
    epochs_executed: int
    stop_reason: str


class GoCLSController:
    """
    Controls how long sleep replay/consolidation continues.

    The controller implements validation-based early stopping:

        replay
          ↓
        student update
          ↓
        validation evaluation
          ↓
        improvement?
          ├── yes → continue
          └── no  → stop after patience is exhausted

    Parameters
    ----------
    replay : SleepReplay
        Sleep replay system that performs Notebook retrieval
        and Student learning.

    patience : int
        Number of consecutive epochs without sufficient
        validation improvement before consolidation stops.

    min_delta : float
        Minimum reduction in validation loss required to count
        as an improvement.

    max_epochs : int
        Hard upper bound on the number of consolidation epochs.
    """

    def __init__(
        self,
        replay: SleepReplay,
        patience: int,
        min_delta: float,
        max_epochs: int,
    ):
        if patience < 0:
            raise ValueError(
                "patience must be non-negative."
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta must be non-negative."
            )

        if max_epochs <= 0:
            raise ValueError(
                "max_epochs must be greater than 0."
            )

        self.replay = replay
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs

    @staticmethod
    def validation_loss(
        student,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> float:
        """
        Evaluate the current Student on held-out validation data.

        This data is NOT used for the replay update.
        """

        if x_validation.ndim != 2:
            raise ValueError(
                "x_validation must be a 2D array."
            )

        if y_validation.ndim != 1:
            raise ValueError(
                "y_validation must be a 1D array."
            )

        if x_validation.shape[0] != y_validation.shape[0]:
            raise ValueError(
                "Validation inputs and targets must "
                "contain the same number of examples."
            )

        output = student.forward(
            x_validation
        )

        return float(
            0.5
            * np.mean(
                (y_validation - output.y_hat) ** 2
            )
        )

    def run(
        self,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> ConsolidationResult:
        """
        Run Go-CLS-style regulated consolidation.

        The validation set is never passed to the learning rule.
        It only determines whether further consolidation is
        beneficial.
        """

        history = []
        validation_losses = []

        best_validation_loss = np.inf
        best_epoch = 0

        epochs_without_improvement = 0

        stop_reason = "max_epochs_reached"

        for epoch in range(1, self.max_epochs + 1):

            # -------------------------------------------------
            # 1. Perform one sleep replay epoch.
            # -------------------------------------------------

            metrics = self.replay.run_epoch()

            history.append(metrics)

            # -------------------------------------------------
            # 2. Evaluate generalization on held-out examples.
            # -------------------------------------------------

            validation_loss = self.validation_loss(
                student=self.replay.student,
                x_validation=x_validation,
                y_validation=y_validation,
            )

            validation_losses.append(
                validation_loss
            )

            # -------------------------------------------------
            # 3. Check for improvement.
            # -------------------------------------------------

            improvement = (
                best_validation_loss
                - validation_loss
            )

            if improvement > self.min_delta:

                best_validation_loss = (
                    validation_loss
                )

                best_epoch = epoch

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            # -------------------------------------------------
            # 4. Stop if consolidation has stopped helping.
            # -------------------------------------------------

            if epochs_without_improvement >= self.patience:
                stop_reason = "validation_patience_exceeded"
                break

        return ConsolidationResult(
            history=history,
            validation_losses=validation_losses,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            epochs_executed=len(history),
            stop_reason=stop_reason,
        )