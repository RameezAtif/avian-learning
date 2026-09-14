from experiments.config import ExperimentConfig
from experiments.runner import run_experiment


RULES = [
    "gradient_descent",
    "chl",
    "qpc",
    "hebbian",
    "anti_hebbian",
]


config = ExperimentConfig(
    num_train_examples=500,
    num_validation_examples=200,
    input_dim=100,
    hidden_dim=100,
    snr=4.0,
    max_epochs=100,
    patience=10,
    replays_per_epoch=32,
    learning_rate=0.001,
    acquisition_epsilon=0.01,
    stability_fraction=0.10,
    stability_windows=5,
    update_w2=False,
    gradient_clip=1.0,
    seed=42,
)


print("===== LEARNING RULE COMPARISON =====")
print()

for rule_name in RULES:

    print(
        f"Running: {rule_name}"
    )

    result = run_experiment(
        learning_rule_name=rule_name,
        config=config,
    )

    print(
        "  Epochs:",
        result.epochs_executed,
    )

    print(
        "  Best validation loss:",
        result.best_validation_loss,
    )

    print(
        "  Acquisition epoch:",
        result.acquisition_epoch,
    )

    print(
        "  Stability:",
        result.stability.stability_variance,
    )

    print()