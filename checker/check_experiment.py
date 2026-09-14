from experiments.config import ExperimentConfig
from experiments.runner import run_experiment


config = ExperimentConfig(
    num_tutor_examples=50,
    num_practice_examples=50,
    num_validation_examples=200,
    input_dim=100,
    hidden_dim=100,
    max_epochs=100,
    patience=10,
    replays_per_epoch=32,
    learning_rate=0.001,
    acquisition_improvement_fraction=0.50,
    stability_fraction=0.10,
    stability_windows=5,
    update_w2=False,
    gradient_clip=1.0,
    seed=42,
)


result = run_experiment(
    learning_rule_name="gradient_descent",
    config=config,
)


print("===== EXPERIMENT =====")

print(
    "Learning rule:",
    result.learning_rule,
)

print(
    "Tutor / practice SNR:",
    result.tutor_snr,
    "/",
    result.practice_snr,
)

print(
    "Seed:",
    result.seed,
)

print(
    "Epochs executed:",
    result.epochs_executed,
)

print(
    "Best validation epoch:",
    result.best_epoch,
)

print(
    "Best validation loss:",
    result.best_validation_loss,
)

print(
    "Acquisition epoch:",
    result.acquisition_epoch,
)

if result.stability is None:
    print("Stability variance: unavailable")
else:
    print("Stability variance:", result.stability.stability_variance)

print()

print("Replay losses:")

for epoch, loss in enumerate(
    result.replay_losses,
    start=1,
):
    print(
        f"{epoch:03d}: {loss:.6f}"
    )
