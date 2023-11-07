import os

import numpy as np
import pandas as pd

from polaris.benchmark import MultiTaskBenchmarkSpecification, SingleTaskBenchmarkSpecification
from polaris.evaluate._metric import Metric
from polaris.evaluate._results import BenchmarkResults
from polaris.utils.types import HubOwner


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


def test_metrics_singletask_reg(tmpdir: str, test_single_task_benchmark: SingleTaskBenchmarkSpecification):
    _, test = test_single_task_benchmark.get_train_test_split()
    predictions = np.random.random(size=test.inputs.shape[0])
    result = test_single_task_benchmark.evaluate(predictions)
    assert isinstance(result.results, pd.DataFrame)
    assert set(result.results.columns) == {"Test set", "Target label", "Metric", "Score"}
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
    result = test_single_task_benchmark_clf.evaluate(predictions)
    for metric in test_single_task_benchmark_clf.metrics:
        assert metric in result.results.Metric.tolist()


def test_metrics_multitask_clf(tmpdir: str, test_multi_task_benchmark_clf: MultiTaskBenchmarkSpecification):
    train, test = test_multi_task_benchmark_clf.get_train_test_split()
    predictions = {
        target_col: np.random.randint(2, size=test.inputs.shape[0]) for target_col in train.target_cols
    }
    result = test_multi_task_benchmark_clf.evaluate(predictions)
    assert isinstance(result.results, pd.DataFrame)
    assert set(result.results.columns) == {"Test set", "Target label", "Metric", "Score"}
    # check the targets and metrics
    assert result.results.shape[0] == len(test_multi_task_benchmark_clf.target_cols) * len(
        test_multi_task_benchmark_clf.metrics
    )
    for metric in test_multi_task_benchmark_clf.metrics:
        assert metric in result.results.Metric.tolist()


def test_metric_direction():
    for metric in Metric:
        assert metric.value.direction in ["min", "max"]
