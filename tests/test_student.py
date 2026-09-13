import numpy as np

from data.teacher import generate_teacher_dataset
from models.student import Student


def test_student_weight_shapes():
    student = Student(
        input_dim=100,
        hidden_dim=100,
        seed=42,
    )

    assert student.W1.shape == (100, 100)
    assert student.W2.shape == (1, 100)


def test_student_reproducibility():
    student_1 = Student(
        input_dim=100,
        hidden_dim=100,
        seed=42,
    )

    student_2 = Student(
        input_dim=100,
        hidden_dim=100,
        seed=42,
    )

    assert np.array_equal(student_1.W1, student_2.W1)
    assert np.array_equal(student_1.W2, student_2.W2)


def test_student_forward_shapes():
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

    assert output.h_ff.shape == (100, 100)
    assert output.y_hat.shape == (100,)


def test_student_single_example():
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

    output = student.forward(dataset.x[0])

    assert output.h_ff.shape == (100,)
    assert np.ndim(output.y_hat) == 0


def test_student_forward_is_deterministic():
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

    output_1 = student.forward(dataset.x)
    output_2 = student.forward(dataset.x)

    assert np.array_equal(
        output_1.h_ff,
        output_2.h_ff,
    )

    assert np.array_equal(
        output_1.y_hat,
        output_2.y_hat,
    )