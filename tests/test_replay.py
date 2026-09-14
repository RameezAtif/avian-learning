import numpy as np

from consolidation.replay import SleepReplay
from data.teacher import generate_teacher_dataset
from learning.plasticity import ContinuousPlasticityRule
from memory.notebook import SparseHopfieldNotebook
from models.student import Student


def create_replay_system():
    dataset = generate_teacher_dataset(
        num_examples=100,
        input_dim=100,
        snr=4,
        seed=42,
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
        x=dataset.x,
        y=dataset.y,
    )

    learning_rule = ContinuousPlasticityRule(
        gamma=1.0,
        eta=0.0,
        learning_rate=0.001,
    )

    replay = SleepReplay(
        student=student,
        notebook=notebook,
        learning_rule=learning_rule,
        replay_cycles=9,
        replays_per_epoch=16,
    )

    return (
        dataset,
        student,
        notebook,
        learning_rule,
        replay,
    )


def test_replay_epoch_returns_metrics():
    _, _, _, _, replay = create_replay_system()

    metrics = replay.run_epoch()

    assert isinstance(metrics.loss, float)
    assert metrics.replayed_memories == 16
    assert 0.0 <= metrics.mean_similarity <= 1.0


def test_replay_changes_w1():
    _, student, _, _, replay = create_replay_system()

    old_w1 = student.W1.copy()

    replay.run_epoch()

    assert not np.array_equal(
        old_w1,
        student.W1,
    )


def test_replay_does_not_change_w2_by_default():
    _, student, _, _, replay = create_replay_system()

    old_w2 = student.W2.copy()

    replay.run_epoch()

    assert np.array_equal(
        old_w2,
        student.W2,
    )


def test_multiple_replay_epochs_produce_history():
    _, _, _, _, replay = create_replay_system()

    history = replay.run(
        epochs=10,
    )

    assert len(history) == 10

    for metrics in history:
        assert metrics.loss >= 0.0
        assert metrics.replayed_memories == 16