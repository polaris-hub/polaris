import numpy as np
import pandas as pd
import pytest

from polaris.benchmark import BenchmarkSpecification
from polaris.dataset import Dataset
from polaris.evaluate._metric import GroupedMetric, Metric


def test_absolute_average_fold_error():
    y_true = np.random.uniform(low=50, high=100, size=200)
    y_pred_1 = y_true + np.random.uniform(low=0, high=5, size=200)
    y_pred_2 = y_true + np.random.uniform(low=5, high=20, size=200)
    y_pred_3 = y_true - 10
    y_zero = np.zeros(shape=200)

    metric = Metric(metric="absolute_average_fold_error")
    # Optimal value
    aafe_0 = metric.fn(y_true=y_true, y_pred=y_true)
    assert aafe_0 == 1

    # small fold change
    aafe_1 = metric.fn(y_true=y_true, y_pred=y_pred_1)
    assert aafe_1 > 1

    # larger fold change
    aafe_2 = metric.fn(y_true=y_true, y_pred=y_pred_2)
    assert aafe_2 > aafe_1

    # undershoot
    aafe_3 = metric.fn(y_true=y_true, y_pred=y_pred_3)
    assert aafe_3 < 1

    # y_true contains zeros
    with pytest.raises(ValueError):
        metric.fn(y_true=y_zero, y_pred=y_pred_3)


def test_grouped_metric():
    metric = GroupedMetric(metric="accuracy", config={"group_by": "group"})

    table = pd.DataFrame({"group": ["a", "b", "b"], "y_true": [1, 1, 1]})
    dataset = Dataset(table=table)
    benchmark = BenchmarkSpecification(
        dataset=dataset,
        metrics=[metric],
        main_metric=metric,
        target_cols=["y_true"],
        input_cols=["group"],
        split=([], [0, 1, 2]),
    )

    result = benchmark.evaluate([1, 0, 0])

    # The global accuracy is only 33%, but because we compute it per group and then average, it's 50%.
    result.results.Score.values[0] == 0.5
