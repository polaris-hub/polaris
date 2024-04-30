import os
import pytest
import numpy as np
import pandas as pd

import polaris as po
from polaris.benchmark import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.evaluate._metric import Metric
from polaris.evaluate._results import BenchmarkResults
from polaris.utils.types import HubOwner
from polaris.dataset import Dataset


def test_result_to_json(tmpdir: str, test_user_owner: HubOwner):
    scores = pd.DataFrame(
        {
            "Test set": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "Target label": ["C", "C", "D", "D", "C", "C", "D", "D"],
            "Metric": ["mean_absolute_error", "accuracy"] * 4,
            "Score": [0.1] * 8,
        }
    )

    result = BenchmarkResults(
        name="test",
        description="Lorem ipsum!",
        tags=["test"],
        user_attributes={"key": "value"},
        owner=test_user_owner,
        results=scores,
        benchmark_name="my-benchmark",
        benchmark_owner=test_user_owner,
        github_url="https://github.com/",
        paper_url="https://chemrxiv.org/",
        contributors=["my-user", "other-user"],
    )

    path = os.path.join(tmpdir, "result.json")
    result.to_json(path)
    BenchmarkResults.from_json(path)
    assert po.__version__ == result.polaris_version


def test_metrics_singletask_reg(tmpdir: str, test_single_task_benchmark: SingleTaskBenchmarkSpecification):
    _, test = test_single_task_benchmark.get_train_test_split()
    predictions = np.random.random(size=test.inputs.shape[0])
    result = test_single_task_benchmark.evaluate(predictions)
    assert isinstance(result.results, pd.DataFrame)
    assert set(result.results.columns) == {
        "Test set",
        "Target label",
        "Metric",
        "Score",
    }
    for metric in test_single_task_benchmark.metrics:
        assert metric in result.results.Metric.tolist()


def test_metrics_multitask_reg(tmpdir: str, test_multi_task_benchmark: MultiTaskBenchmarkSpecification):
    train, test = test_multi_task_benchmark.get_train_test_split()
    predictions = {
        target_col: np.random.random(size=test.inputs.shape[0]) for target_col in train.target_cols
    }
    result = test_multi_task_benchmark.evaluate(predictions)
    for metric in test_multi_task_benchmark.metrics:
        assert metric in result.results.Metric.tolist()


def test_metrics_singletask_clf(
    tmpdir: str, test_single_task_benchmark_clf: SingleTaskBenchmarkSpecification
):
    _, test = test_single_task_benchmark_clf.get_train_test_split()
    predictions = np.random.randint(2, size=test.inputs.shape[0])
    probabilities = np.random.uniform(size=test.inputs.shape[0])
    result = test_single_task_benchmark_clf.evaluate(y_pred=predictions, y_prob=probabilities)
    for metric in test_single_task_benchmark_clf.metrics:
        assert metric in result.results.Metric.tolist()


def test_metrics_singletask_multicls_clf(
    tmpdir: str, test_single_task_benchmark_multi_clf: SingleTaskBenchmarkSpecification
):
    _, test = test_single_task_benchmark_multi_clf.get_train_test_split()
    predictions = np.random.randint(3, size=test.inputs.shape[0])
    probablities = np.random.random(size=(test.inputs.shape[0], 3))
    probablities = probablities / probablities.sum(axis=1, keepdims=True)
    result = test_single_task_benchmark_multi_clf.evaluate(y_pred=predictions, y_prob=probablities)
    for metric in test_single_task_benchmark_multi_clf.metrics:
        assert metric in result.results.Metric.tolist()


def test_metrics_multitask_clf(tmpdir: str, test_multi_task_benchmark_clf: MultiTaskBenchmarkSpecification):
    train, test = test_multi_task_benchmark_clf.get_train_test_split()
    predictions = {
        target_col: np.random.randint(2, size=test.inputs.shape[0]) for target_col in train.target_cols
    }
    probabilities = {
        target_col: np.random.uniform(size=test.inputs.shape[0]) for target_col in train.target_cols
    }
    result = test_multi_task_benchmark_clf.evaluate(y_pred=predictions, y_prob=probabilities)
    assert isinstance(result.results, pd.DataFrame)
    assert set(result.results.columns) == {
        "Test set",
        "Target label",
        "Metric",
        "Score",
    }
    # check the targets and metrics
    assert result.results.shape[0] == len(test_multi_task_benchmark_clf.target_cols) * len(
        test_multi_task_benchmark_clf.metrics
    )
    for metric in test_multi_task_benchmark_clf.metrics:
        assert metric in result.results.Metric.tolist()


def test_metric_direction():
    for metric in Metric:
        assert metric.value.direction in ["min", "max", 1]


def test_absolute_average_fold_error():
    y_true = np.random.uniform(low=50, high=100, size=200)
    y_pred_1 = y_true + np.random.uniform(low=0, high=5, size=200)
    y_pred_2 = y_true + np.random.uniform(low=5, high=20, size=200)
    y_pred_3 = y_true - 10
    y_zero = np.zeros(shape=200)

    # Optimal value
    aafe_0 = Metric.absolute_average_fold_error(y_true=y_true, y_pred=y_true)
    assert aafe_0 == 1

    # small fold change
    aafe_1 = Metric.absolute_average_fold_error(y_true=y_true, y_pred=y_pred_1)
    assert aafe_1 > 1

    # larger fold change
    aafe_2 = Metric.absolute_average_fold_error(y_true=y_true, y_pred=y_pred_2)
    assert aafe_2 > aafe_1

    # undershoot
    aafe_3 = Metric.absolute_average_fold_error(y_true=y_true, y_pred=y_pred_3)
    assert aafe_3 < 1

    # y_true contains zeros
    with pytest.raises(ValueError):
        Metric.absolute_average_fold_error(y_true=y_zero, y_pred=y_pred_3)


def test_metric_y_types(
    tmpdir: str, test_single_task_benchmark_clf: SingleTaskBenchmarkSpecification, test_data: Dataset
):
    # here we use train split for testing purpose.
    _, test = test_single_task_benchmark_clf.get_train_test_split()
    predictions = np.random.randint(2, size=test.inputs.shape[0])
    probabilities = np.random.uniform(size=test.inputs.shape[0])
    test_y = test_data.loc[test.indices, "CLASS_expt"]

    # If y_pred is None and y_prob is None, an error is thrown.
    with pytest.raises(ValueError, match="Neither `y_pred` nor `y_prob` is specified."):
        test_single_task_benchmark_clf.evaluate()

    # If y_type == "y_pred" and y_pred is None, an error is thrown.
    with pytest.raises(ValueError, match="Metric.accuracy requires `y_pred` input"):
        test_single_task_benchmark_clf.metrics = [Metric.accuracy]
        test_single_task_benchmark_clf.evaluate(y_prob=probabilities)

    # If y_type != "y_pred" and y_prob is None, an error is thrown.
    with pytest.raises(ValueError, match="Metric.roc_auc requires `y_prob` input"):
        test_single_task_benchmark_clf.metrics = [Metric.roc_auc]
        test_single_task_benchmark_clf.evaluate(y_pred=predictions)

    # If y_type != "y_pred" and y_pred is not None and y_prob is not None, it uses y_prob as expected!
    test_single_task_benchmark_clf.metrics = [Metric.roc_auc]
    result = test_single_task_benchmark_clf.evaluate(y_pred=predictions, y_prob=probabilities)
    assert result.results.Score.values[0] == Metric.roc_auc.fn(y_true=test_y, y_score=probabilities)

    # If y_type == "y_pred" and y_pred is not None and y_prob is not None, it uses y_pred as expected!
    test_single_task_benchmark_clf.metrics = [Metric.f1]
    result = test_single_task_benchmark_clf.evaluate(y_pred=predictions, y_prob=probabilities)
    assert result.results.Score.values[0] == Metric.f1.fn(y_true=test_y, y_pred=predictions)
