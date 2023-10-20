import os
import numpy as np
import pandas as pd
from polaris.benchmark import SingleTaskBenchmarkSpecification, MultiTaskBenchmarkSpecification
from polaris.evaluate._results import BenchmarkResults
from polaris.utils.types import HubOwner


def test_result_to_json(tmpdir: str, test_user_owner: HubOwner):
    scores = pd.DataFrame({"Test set": ["A"], "Target label": "B", "Metric": "C", "Score": 0.1})

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


def test_metrics_singletask_reg(
    tmpdir: str, test_single_task_benchmark_reg: SingleTaskBenchmarkSpecification
):
    train, test = test_single_task_benchmark_reg.get_train_test_split()
    predictions = np.random.random(size=test.inputs.shape[0])
    result = test_single_task_benchmark_reg.evaluate(predictions)
    path = os.path.join(tmpdir, "result_singletask_reg.json")
    result.to_json(path)
    BenchmarkResults.from_json(path)


def test_metrics_multitask_reg(tmpdir: str, test_multi_task_benchmark_reg: MultiTaskBenchmarkSpecification):
    train, test = test_multi_task_benchmark_reg.get_train_test_split()
    predictions = {
        target_col: np.random.random(size=test.inputs.shape[0]) for target_col in train.target_cols
    }
    result = test_multi_task_benchmark_reg.evaluate(predictions)
    path = os.path.join(tmpdir, "result_multitask_reg.json")
    result.to_json(path)
    BenchmarkResults.from_json(path)


def test_metrics_singletask_clf(
    tmpdir: str, test_single_task_benchmark_clf: SingleTaskBenchmarkSpecification
):
    train, test = test_single_task_benchmark_clf.get_train_test_split()
    predictions = np.random.randint(2, size=test.inputs.shape[0])
    result = test_single_task_benchmark_clf.evaluate(predictions)
    path = os.path.join(tmpdir, "result_singletask_clf.json")
    result.to_json(path)
    BenchmarkResults.from_json(path)


def test_metrics_multitask_clf(tmpdir: str, test_multi_task_benchmark_clf: MultiTaskBenchmarkSpecification):
    train, test = test_multi_task_benchmark_clf.get_train_test_split()
    predictions = {
        target_col: np.random.randint(2, size=test.inputs.shape[0]) for target_col in train.target_cols
    }
    result = test_multi_task_benchmark_clf.evaluate(predictions)
    path = os.path.join(tmpdir, "result_multitask_clf.json")
    result.to_json(path)
    BenchmarkResults.from_json(path)
