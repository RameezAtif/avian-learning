from data.teacher import generate_teacher_dataset
from models.student import Student

from learning.gradient_descent import (
    apply_gradient_descent,
    calculate_gradients,
    mean_squared_error,
)


dataset = generate_teacher_dataset(
    num_examples=10000,
    input_dim=100,
    snr=4,
    seed=42,
)

student = Student(
    input_dim=100,
    hidden_dim=100,
    seed=42,
)

# ---------------------------------------------------------
# Initial forward pass
# ---------------------------------------------------------

output_before = student.forward(dataset.x)

loss_before = mean_squared_error(
    dataset.y,
    output_before.y_hat,
)

# ---------------------------------------------------------
# Calculate gradients
# ---------------------------------------------------------

gradients = calculate_gradients(
    x=dataset.x,
    y=dataset.y,
    h=output_before.h_ff,
    y_hat=output_before.y_hat,
    w2=student.W2,
)

# ---------------------------------------------------------
# Apply one gradient update
# ---------------------------------------------------------

student.W1, student.W2 = apply_gradient_descent(
    w1=student.W1,
    w2=student.W2,
    gradients=gradients,
    learning_rate=0.01,
    update_w2=True,
)

# ---------------------------------------------------------
# Forward pass after update
# ---------------------------------------------------------

output_after = student.forward(dataset.x)

loss_after = mean_squared_error(
    dataset.y,
    output_after.y_hat,
)

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("===== GRADIENT DESCENT CHECK =====")

print("Initial loss:", loss_before)
print("Loss after one update:", loss_after)

print()

print("W1 gradient shape:", gradients.grad_w1.shape)
print("W2 gradient shape:", gradients.grad_w2.shape)

print()

print("Loss decreased:", loss_after < loss_before)