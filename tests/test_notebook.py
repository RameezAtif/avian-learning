import numpy as np

from memory.notebook import SparseHopfieldNotebook


def create_notebook():
    return SparseHopfieldNotebook(
        notebook_dim=100,
        sparsity=0.05,
        seed=42,
    )


def test_notebook_starts_empty():
    notebook = create_notebook()

    assert len(notebook) == 0


def test_sparse_pattern_has_correct_number_of_active_units():
    notebook = create_notebook()

    pattern = notebook._create_sparse_pattern()

    assert pattern.shape == (100,)
    assert np.sum(pattern) == 5


def test_encode_stores_memory():
    notebook = create_notebook()

    x = np.random.default_rng(42).normal(
        size=100,
    )

    y = np.asarray(0.5)

    memory = notebook.encode(
        x=x,
        y=y,
    )

    assert len(notebook) == 1

    assert memory.pattern.shape == (
        notebook.notebook_dim,
    )

    assert np.sum(memory.pattern) == (
        notebook.active_units
    )

    assert np.array_equal(
        memory.x,
        x,
    )

    assert np.array_equal(
        memory.y,
        y,
    )


def test_encode_batch_stores_all_examples():
    notebook = create_notebook()

    rng = np.random.default_rng(42)

    x = rng.normal(
        size=(10, 100),
    )

    y = rng.normal(
        size=10,
    )

    notebook.encode_batch(
        x=x,
        y=y,
    )

    assert len(notebook) == 10


def test_retrieval_returns_valid_memory():
    notebook = create_notebook()

    rng = np.random.default_rng(42)

    x = rng.normal(
        size=(10, 100),
    )

    y = rng.normal(
        size=10,
    )

    notebook.encode_batch(
        x=x,
        y=y,
    )

    result = notebook.retrieve(
        cycles=9,
    )

    assert result.pattern.shape == (
        notebook.notebook_dim,
    )

    assert np.sum(result.pattern) == (
        notebook.active_units
    )

    assert 0 <= result.memory_index < 10

    assert 0.0 <= result.similarity <= 1.0

    assert result.x.shape == (100,)

    assert np.ndim(result.y) == 0


def test_replay_batch_returns_requested_number():
    notebook = create_notebook()

    rng = np.random.default_rng(42)

    x = rng.normal(
        size=(10, 100),
    )

    y = rng.normal(
        size=10,
    )

    notebook.encode_batch(
        x=x,
        y=y,
    )

    results = notebook.replay_batch(
        num_replays=20,
        cycles=9,
    )

    assert len(results) == 20