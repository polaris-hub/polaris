import numpy as np
import pytest

from polaris.evaluate._metric import Metric


def test_absolute_average_fold_error():
    y_true = np.random.uniform(low=50, high=100, size=200)
    y_pred_1 = y_true + np.random.uniform(low=0, high=5, size=200)
    y_pred_2 = y_true + np.random.uniform(low=5, high=20, size=200)
    y_pred_3 = y_true - 10
    y_zero = np.zeros(shape=200)

    # Optimal value
    aafe_0 = Metric.absolute_average_fold_error.fn(y_true=y_true, y_pred=y_true)
    assert aafe_0 == 1

    # small fold change
    aafe_1 = Metric.absolute_average_fold_error.fn(y_true=y_true, y_pred=y_pred_1)
    assert aafe_1 > 1

    # larger fold change
    aafe_2 = Metric.absolute_average_fold_error.fn(y_true=y_true, y_pred=y_pred_2)
    assert aafe_2 > aafe_1

    # undershoot
    aafe_3 = Metric.absolute_average_fold_error.fn(y_true=y_true, y_pred=y_pred_3)
    assert aafe_3 < 1

    # y_true contains zeros
    with pytest.raises(ValueError):
        Metric.absolute_average_fold_error.fn(y_true=y_zero, y_pred=y_pred_3)
