from data.teacher import generate_teacher_dataset
from learning.plasticity import ContinuousPlasticityRule
from memory.notebook import SparseHopfieldNotebook
from models.student import Student
from consolidation.replay import SleepReplay


# ---------------------------------------------------------
# 1. Generate teacher experiences
# ---------------------------------------------------------

dataset = generate_teacher_dataset(
    num_examples=1000,
    input_dim=100,
    snr=4,
    seed=42,
)


# ---------------------------------------------------------
# 2. Create Student
# ---------------------------------------------------------

student = Student(
    input_dim=100,
    hidden_dim=100,
    seed=42,
)


# ---------------------------------------------------------
# 3. Create Notebook
# ---------------------------------------------------------

notebook = SparseHopfieldNotebook(
    notebook_dim=500,
    sparsity=0.05,
    seed=42,
)


# ---------------------------------------------------------
# 4. Encode teacher experiences
# ---------------------------------------------------------

notebook.encode_batch(
    x=dataset.x,
    y=dataset.y,
)


# ---------------------------------------------------------
# 5. Select learning rule
#
# CHL for this test.
# ---------------------------------------------------------

learning_rule = ContinuousPlasticityRule(
    gamma=1.0,
    eta=0.0,
    learning_rate=0.001,
)


# ---------------------------------------------------------
# 6. Create sleep replay system
# ---------------------------------------------------------

replay = SleepReplay(
    student=student,
    notebook=notebook,
    learning_rule=learning_rule,
    replay_cycles=9,
    replays_per_epoch=32,
)


# ---------------------------------------------------------
# 7. Run replay
# ---------------------------------------------------------

history = replay.run(
    epochs=20,
)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------

print("===== SLEEP REPLAY =====")

print(
    "Stored Notebook memories:",
    len(notebook),
)

print(
    "Replay epochs:",
    len(history),
)

print()

for epoch, metrics in enumerate(history, start=1):

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {metrics.loss:.6f} | "
        f"Replay memories: "
        f"{metrics.replayed_memories} | "
        f"Similarity: "
        f"{metrics.mean_similarity:.4f}"
    )