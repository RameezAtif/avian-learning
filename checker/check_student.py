import numpy as np

from data.teacher import generate_teacher_dataset
from models.student import Student


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

output = student.forward(dataset.x)


# ---------------------------------------------------------
# Print basic information
# ---------------------------------------------------------

print("===== DATASET =====")

print("Input shape:", dataset.x.shape)
print("Target shape:", dataset.y.shape)

print()

print("===== STUDENT WEIGHTS =====")

print("W1 shape:", student.W1.shape)
print("W2 shape:", student.W2.shape)

print()

print("===== STUDENT OUTPUT =====")

print("Hidden state shape:", output.h_ff.shape)
print("Prediction shape:", output.y_hat.shape)

print()

print("===== STATISTICS =====")

print("Mean of hidden state:", np.mean(output.h_ff))
print("Variance of hidden state:", np.var(output.h_ff))

print("Mean of predictions:", np.mean(output.y_hat))
print("Variance of predictions:", np.var(output.y_hat))

print()

print("===== SAMPLE VALUES =====")

print("First input:")
print(dataset.x[0])

print()

print("First hidden state:")
print(output.h_ff[0])

print()

print("First teacher target:")
print(dataset.y[0])

print()

print("First student prediction:")
print(output.y_hat[0])