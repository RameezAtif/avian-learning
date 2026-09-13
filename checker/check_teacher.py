import numpy as np
import matplotlib.pyplot as plt

from data.teacher import generate_teacher_dataset


dataset = generate_teacher_dataset(
    num_examples=10000,
    input_dim=100,
    snr=4,
    seed=42,
)

# ---------------------------------------------------------
# Theoretical variances implied by the requested SNR
# ---------------------------------------------------------

if np.isinf(dataset.snr):
    theoretical_signal_variance = 1.0
    theoretical_noise_variance = 0.0
else:
    theoretical_signal_variance = dataset.snr / (dataset.snr + 1.0)
    theoretical_noise_variance = 1.0 / (dataset.snr + 1.0)

# ---------------------------------------------------------
# Print basic information
# ---------------------------------------------------------

print("===== DATASET INFORMATION =====")

print("Input shape:", dataset.x.shape)
print("Output shape:", dataset.y.shape)

print()

print("Requested SNR:", dataset.snr)

print()
print("===== THEORETICAL VALUES =====")

print(
    "Theoretical signal variance:",
    theoretical_signal_variance,
)

print(
    "Theoretical noise variance:",
    theoretical_noise_variance,
)

print()
print("===== EMPIRICAL VALUES =====")

print(
    "Empirical signal variance:",
    dataset.signal_variance,
)

print(
    "Empirical noise variance:",
    dataset.noise_variance,
)

empirical_snr = (
    dataset.signal_variance / dataset.noise_variance
)

print("Empirical SNR:", empirical_snr)

print()
print("===== INPUT STATISTICS =====")

print("Mean of x:", np.mean(dataset.x))
print("Variance of x:", np.var(dataset.x))

print()
print("===== OUTPUT STATISTICS =====")

print("Mean of y:", np.mean(dataset.y))
print("Variance of y:", np.var(dataset.y))

# ---------------------------------------------------------
# Plot the teacher outputs
# ---------------------------------------------------------

plt.hist(dataset.y, bins=40)

plt.xlabel("Teacher output y")
plt.ylabel("Frequency")
plt.title("Synthetic Teacher Outputs")

plt.show()