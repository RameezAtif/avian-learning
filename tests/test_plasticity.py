import numpy as np

from data.teacher import generate_teacher_dataset
from learning.plasticity import ContinuousPlasticityRule
from models.student import Student


def create_test_case():
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

    return dataset, student, output


def test_chl_produces_nonzero_update():
    dataset, student, output = create_test_case()

    rule = ContinuousPlasticityRule(
        gamma=1.0,
        eta=0.0,
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

    assert update.delta_w1.shape == student.W1.shape
    assert np.any(update.delta_w1 != 0.0)


def test_quasi_predictive_coding_reverses_feedback():
    dataset, student, output = create_test_case()

    chl = ContinuousPlasticityRule(
        gamma=1.0,
        eta=0.0,
        learning_rate=1.0,
    )

    qpc = ContinuousPlasticityRule(
        gamma=-1.0,
        eta=0.0,
        learning_rate=1.0,
    )

    chl_update = chl.calculate_update(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w1=student.W1,
        w2=student.W2,
    )

    qpc_update = qpc.calculate_update(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w1=student.W1,
        w2=student.W2,
    )

    assert np.allclose(
        qpc_update.delta_w1,
        -chl_update.delta_w1,
    )


def test_eta_zero_removes_hebbian_component():
    dataset, student, output = create_test_case()

    rule_a = ContinuousPlasticityRule(
        gamma=0.0,
        eta=0.0,
        learning_rate=1.0,
    )

    update = rule_a.calculate_update(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w1=student.W1,
        w2=student.W2,
    )

    assert np.allclose(
        update.delta_w1,
        0.0,
    )


def test_pure_hebbian_has_no_feedback_component():
    dataset, student, output = create_test_case()

    rule = ContinuousPlasticityRule(
        gamma=0.0,
        eta=0.1,
        learning_rate=1.0,
    )

    update = rule.calculate_update(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w1=student.W1,
        w2=student.W2,
    )

    assert update.delta_w1.shape == student.W1.shape
    assert np.any(update.delta_w1 != 0.0)


def test_anti_hebbian_has_no_feedback_component():
    dataset, student, output = create_test_case()

    rule = ContinuousPlasticityRule(
        gamma=0.0,
        eta=-0.1,
        learning_rate=1.0,
    )

    update = rule.calculate_update(
        x=dataset.x,
        y=dataset.y,
        h=output.h_ff,
        y_hat=output.y_hat,
        w1=student.W1,
        w2=student.W2,
    )

    assert update.delta_w1.shape == student.W1.shape
    assert np.any(update.delta_w1 != 0.0)


def test_w2_is_unchanged_for_now():
    dataset, student, output = create_test_case()

    rule = ContinuousPlasticityRule(
        gamma=1.0,
        eta=0.0,
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

    assert np.allclose(
        update.delta_w2,
        0.0,
    )


def test_error_driven_rule_can_update_w2_when_enabled():
    dataset, student, output = create_test_case()
    rule = ContinuousPlasticityRule(
        gamma=1.0, eta=0.0, learning_rate=0.01, update_w2=True,
    )
    update = rule.calculate_update(
        x=dataset.x, y=dataset.y, h=output.h_ff, y_hat=output.y_hat,
        w1=student.W1, w2=student.W2,
    )
    assert np.any(update.delta_w2 != 0.0)
