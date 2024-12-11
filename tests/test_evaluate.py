from pathlib import Path

import datamol as dm
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

import polaris as po
from polaris.benchmark import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset import DatasetV1
from polaris.evaluate._metric import DEFAULT_METRICS, Metric
from polaris.evaluate._results import BenchmarkResults
from polaris.utils.types import HubOwner


def test_result_to_json(tmp_path: Path, test_user_owner: HubOwner):
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

    path = str(tmp_path / "result.json")
    result.to_json(path)

    BenchmarkResults.from_json(path)
    assert po.__version__ == result.polaris_version


def test_metrics_singletask_reg(test_single_task_benchmark: SingleTaskBenchmarkSpecification):
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
        assert metric.name in result.results.Metric.tolist()


def test_metrics_multitask_reg(test_multi_task_benchmark: MultiTaskBenchmarkSpecification):
    train, test = test_multi_task_benchmark.get_train_test_split()
    predictions = {
        target_col: np.random.random(size=test.inputs.shape[0]) for target_col in train.target_cols
    }
    result = test_multi_task_benchmark.evaluate(predictions)
    for metric in test_multi_task_benchmark.metrics:
        assert metric.name in result.results.Metric.tolist()


def test_metrics_singletask_clf(test_single_task_benchmark_clf: SingleTaskBenchmarkSpecification):
    _, test = test_single_task_benchmark_clf.get_train_test_split()
    predictions = np.random.randint(2, size=test.inputs.shape[0])
    probabilities = np.random.uniform(size=test.inputs.shape[0])
    result = test_single_task_benchmark_clf.evaluate(y_pred=predictions, y_prob=probabilities)
    for metric in test_single_task_benchmark_clf.metrics:
        assert metric.name in result.results.Metric.tolist()


def test_metrics_singletask_multicls_clf(
    test_single_task_benchmark_multi_clf: SingleTaskBenchmarkSpecification,
):
    _, test = test_single_task_benchmark_multi_clf.get_train_test_split()
    predictions = np.random.randint(3, size=test.inputs.shape[0])
    probablities = np.random.random(size=(test.inputs.shape[0], 3))
    probablities = probablities / probablities.sum(axis=1, keepdims=True)
    result = test_single_task_benchmark_multi_clf.evaluate(y_pred=predictions, y_prob=probablities)
    for metric in test_single_task_benchmark_multi_clf.metrics:
        assert metric.name in result.results.Metric.tolist()


def test_metrics_multitask_clf(test_multi_task_benchmark_clf: MultiTaskBenchmarkSpecification):
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
        assert metric.name in result.results.Metric.tolist()


def test_metric_direction():
    for metric_info in DEFAULT_METRICS.values():
        assert metric_info.direction in ["min", "max", 1]


def test_metric_y_types(
    test_single_task_benchmark_clf: SingleTaskBenchmarkSpecification, test_data: DatasetV1
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
    with pytest.raises(ValueError, match="Metric requires `y_pred` input"):
        test_single_task_benchmark_clf.metrics = [Metric(label="accuracy")]
        test_single_task_benchmark_clf.evaluate(y_prob=probabilities)

    # If y_type != "y_pred" and y_prob is None, an error is thrown.
    with pytest.raises(ValueError, match="Metric requires `y_prob` input"):
        test_single_task_benchmark_clf.metrics = [Metric(label="roc_auc")]
        test_single_task_benchmark_clf.evaluate(y_pred=predictions)

    # If y_type != "y_pred" and y_prob is None, an error is thrown.
    with pytest.raises(ValueError, match="Metric requires `y_prob` input"):
        test_single_task_benchmark_clf.metrics = [Metric(label="pr_auc")]
        test_single_task_benchmark_clf.evaluate(y_pred=predictions)

    # If y_type != "y_pred" and y_pred is not None and y_prob is not None, it uses y_prob as expected!
    test_single_task_benchmark_clf.metrics = [Metric(label="roc_auc")]
    result = test_single_task_benchmark_clf.evaluate(y_pred=predictions, y_prob=probabilities)
    assert result.results.Score.values[0] == roc_auc_score(y_true=test_y, y_score=probabilities)

    # If y_type == "y_pred" and y_pred is not None and y_prob is not None, it uses y_pred as expected!
    test_single_task_benchmark_clf.metrics = [Metric(label="f1")]
    result = test_single_task_benchmark_clf.evaluate(y_pred=predictions, y_prob=probabilities)
    assert result.results.Score.values[0] == f1_score(y_true=test_y, y_pred=predictions)


def test_metrics_docking(test_docking_benchmark: SingleTaskBenchmarkSpecification, caffeine, ibuprofen):
    _, test = test_docking_benchmark.get_train_test_split()

    predictions = np.array([caffeine, ibuprofen])
    result = test_docking_benchmark.evaluate(y_pred=predictions)

    for metric in test_docking_benchmark.metrics:
        assert metric.name in result.results.Metric.tolist()

    # sanity check
    assert result.results.Score.values[0] == 1

    conf_caffeine = dm.conformers.generate(mol=caffeine, n_confs=1, random_seed=333)
    conf_ibuprofen = dm.conformers.generate(mol=ibuprofen, n_confs=1, random_seed=333)
    predictions = np.array([conf_caffeine, conf_ibuprofen])
    result = test_docking_benchmark.evaluate(y_pred=predictions)
    assert result.results.Score.values[0] == 0
