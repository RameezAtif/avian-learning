import numpy as np

from data.teacher import generate_teacher_dataset
from memory.notebook import SparseHopfieldNotebook


# ---------------------------------------------------------
# Generate teacher experiences
# ---------------------------------------------------------

dataset = generate_teacher_dataset(
    num_examples=100,
    input_dim=100,
    snr=4,
    seed=42,
)


# ---------------------------------------------------------
# Create Notebook
# ---------------------------------------------------------

notebook = SparseHopfieldNotebook(
    notebook_dim=500,
    sparsity=0.05,
    seed=42,
)


# ---------------------------------------------------------
# Encode experiences
# ---------------------------------------------------------

notebook.encode_batch(
    x=dataset.x,
    y=dataset.y,
)


print("===== NOTEBOOK INFORMATION =====")

print(
    "Notebook size:",
    notebook.notebook_dim,
)

print(
    "Sparsity:",
    notebook.sparsity,
)

print(
    "Active units per memory:",
    notebook.active_units,
)

print(
    "Stored memories:",
    len(notebook),
)

print()

# ---------------------------------------------------------
# Inspect one stored memory
# ---------------------------------------------------------

memory = notebook.memories[0]

print("===== STORED MEMORY =====")

print(
    "Pattern shape:",
    memory.pattern.shape,
)

print(
    "Active units:",
    np.sum(memory.pattern),
)

print(
    "Stored x shape:",
    memory.x.shape,
)

print(
    "Stored y:",
    memory.y,
)

print()

# ---------------------------------------------------------
# Replay
# ---------------------------------------------------------

result = notebook.retrieve(
    cycles=9,
)

print("===== REPLAY RESULT =====")

print(
    "Retrieved memory index:",
    result.memory_index,
)

print(
    "Similarity:",
    result.similarity,
)

print(
    "Retrieved x shape:",
    result.x.shape,
)

print(
    "Retrieved y:",
    result.y,
)

print(
    "Retrieved pattern active units:",
    np.sum(result.pattern),
)

print()

# ---------------------------------------------------------
# Replay many memories
# ---------------------------------------------------------

results = notebook.replay_batch(
    num_replays=20,
    cycles=9,
)

indices = [
    result.memory_index
    for result in results
]

print("===== REPLAY BATCH =====")

print(
    "Number of replays:",
    len(results),
)

print(
    "Retrieved memory indices:",
    indices,
)

print(
    "Unique memories retrieved:",
    len(set(indices)),
)