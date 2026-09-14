import numpy as np

from consolidation.controller import GoCLSController
from consolidation.replay import SleepReplay
from data.teacher import generate_teacher_dataset
from learning.plasticity import ContinuousPlasticityRule
from memory.notebook import SparseHopfieldNotebook
from models.student import Student


def create_system():
    train = generate_teacher_dataset(
        num_examples=80,
        input_dim=100,
        snr=4,
        seed=42,
    )

    validation = generate_teacher_dataset(
        num_examples=20,
        input_dim=100,
        snr=4,
        seed=123,
    )

    # Use the same teacher mapping for validation in a
    # controlled test setup.
    validation_weights = train.teacher_weights

    validation_signal = (
        validation.x @ validation_weights
    )

    validation.y = (
        validation_signal
        + validation.noise
    )

    student = Student(
        input_dim=100,
        hidden_dim=100,
        seed=42,
    )

    notebook = SparseHopfieldNotebook(
        notebook_dim=500,
        sparsity=0.05,
        seed=42,
    )

    notebook.encode_batch(
        x=train.x,
        y=train.y,
    )

    rule = ContinuousPlasticityRule(
        gamma=1.0,
        eta=0.0,
        learning_rate=0.001,
    )

    replay = SleepReplay(
        student=student,
        notebook=notebook,
        learning_rule=rule,
        replay_cycles=9,
        replays_per_epoch=10,
    )

    controller = GoCLSController(
        replay=replay,
        patience=3,
        min_delta=1e-8,
        max_epochs=20,
    )

    return (
        train,
        validation,
        student,
        notebook,
        replay,
        controller,
    )


def test_validation_loss_is_nonnegative():
    (
        _,
        validation,
        student,
        _,
        _,
        controller,
    ) = create_system()

    loss = controller.validation_loss(
        student=student,
        x_validation=validation.x,
        y_validation=validation.y,
    )

    assert loss >= 0.0


def test_controller_returns_result():
    (
        _,
        validation,
        _,
        _,
        _,
        controller,
    ) = create_system()

    result = controller.run(
        x_validation=validation.x,
        y_validation=validation.y,
    )

    assert result.epochs_executed > 0
    assert result.epochs_executed <= 20

    assert len(result.history) == (
        result.epochs_executed
    )

    assert len(result.validation_losses) == (
        result.epochs_executed
    )


def test_validation_loss_history_is_finite():
    (
        _,
        validation,
        _,
        _,
        _,
        controller,
    ) = create_system()

    result = controller.run(
        x_validation=validation.x,
        y_validation=validation.y,
    )

    assert all(
        np.isfinite(loss)
        for loss in result.validation_losses
    )


def test_best_epoch_is_valid():
    (
        _,
        validation,
        _,
        _,
        _,
        controller,
    ) = create_system()

    result = controller.run(
        x_validation=validation.x,
        y_validation=validation.y,
    )

    assert (
        0 <= result.best_epoch
        <= result.epochs_executed
    )