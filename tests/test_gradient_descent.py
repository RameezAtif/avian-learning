import numpy as np

from data.teacher import generate_teacher_dataset
from models.student import Student
from learning.gradient_descent import (
    apply_gradient_descent,
    calculate_gradients,
    mean_squared_error,
)


def test_mean_squared_error_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.0, 2.0, 3.0])

    loss = mean_squared_error(y, y_hat)

    assert loss == 0.0


def test_gradient_shapes():
    dataset = generate_teacher_dataset(
        num_examples=100,
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

    gradients = calculate_gradients(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w2=student.W2,
    )

    assert gradients.grad_w1.shape == student.W1.shape
    assert gradients.grad_w2.shape == student.W2.shape


def test_gradient_descent_changes_w1():
    dataset = generate_teacher_dataset(
        num_examples=100,
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

    gradients = calculate_gradients(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w2=student.W2,
    )

    old_w1 = student.W1.copy()

    new_w1, new_w2 = apply_gradient_descent(
        w1=student.W1,
        w2=student.W2,
        gradients=gradients,
        learning_rate=0.01,
        update_w2=True,
    )

    assert not np.array_equal(old_w1, new_w1)


def test_gradient_descent_reduces_loss_for_one_update():
    dataset = generate_teacher_dataset(
        num_examples=1000,
        input_dim=100,
        snr=4,
        seed=42,
    )

    student = Student(
        input_dim=100,
        hidden_dim=100,
        seed=42,
    )

    # Initial prediction.
    output_before = student.forward(dataset.x)

    loss_before = mean_squared_error(
        dataset.y,
        output_before.y_hat,
    )

    # Calculate gradient.
    gradients = calculate_gradients(
        x=dataset.x,
        y=dataset.y,
        h=output_before.h_ff,
        y_hat=output_before.y_hat,
        w2=student.W2,
    )

    # Apply update.
    student.W1, student.W2 = apply_gradient_descent(
        w1=student.W1,
        w2=student.W2,
        gradients=gradients,
        learning_rate=0.01,
        update_w2=True,
    )

    # Prediction after update.
    output_after = student.forward(dataset.x)

    loss_after = mean_squared_error(
        dataset.y,
        output_after.y_hat,
    )

    assert loss_after < loss_before


def test_w2_can_be_kept_fixed():
    dataset = generate_teacher_dataset(
        num_examples=100,
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

    gradients = calculate_gradients(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w2=student.W2,
    )

    old_w2 = student.W2.copy()

    _, new_w2 = apply_gradient_descent(
        w1=student.W1,
        w2=student.W2,
        gradients=gradients,
        learning_rate=0.01,
        update_w2=False,
    )

    assert np.array_equal(old_w2, new_w2)