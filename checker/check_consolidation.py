from data.teacher import generate_teacher_dataset
from models.student import Student

from memory.notebook import SparseHopfieldNotebook

from learning.plasticity import ContinuousPlasticityRule
from consolidation.replay import SleepReplay
from consolidation.controller import GoCLSController


# ---------------------------------------------------------
# 1. Generate training experiences
# ---------------------------------------------------------

train = generate_teacher_dataset(
    num_examples=100,
    input_dim=100,
    snr=4,
    seed=42,
)


# ---------------------------------------------------------
# 2. Generate validation experiences
# ---------------------------------------------------------

validation = generate_teacher_dataset(
    num_examples=100,
    input_dim=100,
    snr=4,
    seed=123,
)

# Use the SAME teacher as the training set.
validation_signal = (
    validation.x @ train.teacher_weights
)

validation.y = (
    validation_signal
    + validation.noise
)


# ---------------------------------------------------------
# 3. Create Student
# ---------------------------------------------------------

student = Student(
    input_dim=100,
    hidden_dim=100,
    seed=42,
)


# ---------------------------------------------------------
# 4. Create Notebook
# ---------------------------------------------------------

notebook = SparseHopfieldNotebook(
    notebook_dim=500,
    sparsity=0.05,
    seed=42,
)

notebook.encode_batch(
    x=train.x,
    y=train.y,
)


# ---------------------------------------------------------
# 5. Select learning rule
# ---------------------------------------------------------

learning_rule = ContinuousPlasticityRule(
    gamma=1.0,
    eta=0.0,
    learning_rate=0.001,
)


# ---------------------------------------------------------
# 6. Create replay system
# ---------------------------------------------------------

replay = SleepReplay(
    student=student,
    notebook=notebook,
    learning_rule=learning_rule,
    replay_cycles=9,
    replays_per_epoch=32,
)


# ---------------------------------------------------------
# 7. Create Go-CLS controller
# ---------------------------------------------------------

controller = GoCLSController(
    replay=replay,
    patience=5,
    min_delta=1e-6,
    max_epochs=100,
)


# ---------------------------------------------------------
# 8. Run regulated consolidation
# ---------------------------------------------------------

result = controller.run(
    x_validation=validation.x,
    y_validation=validation.y,
)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------

print("===== Go-CLS CONSOLIDATION =====")

print(
    "Training memories:",
    len(notebook),
)

print(
    "Epochs executed:",
    result.epochs_executed,
)

print(
    "Best epoch:",
    result.best_epoch,
)

print(
    "Best validation loss:",
    result.best_validation_loss,
)

print()

print("===== VALIDATION HISTORY =====")

for epoch, loss in enumerate(
    result.validation_losses,
    start=1,
):
    print(
        f"Epoch {epoch:03d} | "
        f"Validation loss: {loss:.6f}"
    )