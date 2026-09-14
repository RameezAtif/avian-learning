import numpy as np

from data.teacher import generate_teacher_dataset
from learning.plasticity import ContinuousPlasticityRule
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


learning_rules = {
    "CHL": {
        "gamma": 1.0,
        "eta": 0.0,
    },
    "Quasi-Predictive Coding": {
        "gamma": -1.0,
        "eta": 0.0,
    },
    "Pure Hebbian": {
        "gamma": 0.0,
        "eta": 0.1,
    },
    "Anti-Hebbian": {
        "gamma": 0.0,
        "eta": -0.1,
    },
}


print("===== CONTINUOUS PLASTICITY CHECK =====")
print()

for name, parameters in learning_rules.items():

    rule = ContinuousPlasticityRule(
        gamma=parameters["gamma"],
        eta=parameters["eta"],
        learning_rate=0.01,
    )

    update = rule.calculate_update(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w1=student.W1,
        w2=student.W2,
    )

    print(name)

    print(
        "  gamma:",
        parameters["gamma"],
    )

    print(
        "  eta:",
        parameters["eta"],
    )

    print(
        "  W1 update shape:",
        update.delta_w1.shape,
    )

    print(
        "  W1 update norm:",
        np.linalg.norm(update.delta_w1),
    )

    print(
        "  W2 update norm:",
        np.linalg.norm(update.delta_w2),
    )

    print()