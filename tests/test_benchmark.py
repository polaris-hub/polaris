import pytest

from polaris.dataset import Benchmark, Task
from polaris.utils.errors import InvalidBenchmarkError


def test_single_task_benchmark_split_verification(test_single_task_benchmark):
    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "dataset": test_single_task_benchmark.dataset,
        "target_cols": test_single_task_benchmark.target_cols,
        "input_cols": test_single_task_benchmark.input_cols,
        "metrics": test_single_task_benchmark._metrics,
    }

    train_split = test_single_task_benchmark._split[0]
    test_split = test_single_task_benchmark._split[1]

    # One or more empty partitions
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split,), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, []), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, {"test": []}), **default_kwargs)
    # Non-exclusive partitions
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, test_split + train_split[:1]), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, {"test1": test_split, "test2": train_split[:1]}), **default_kwargs)
    # Invalid indices
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(
            split=(train_split + [len(test_single_task_benchmark.dataset)], test_split), **default_kwargs
        )
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split + [-1], test_split), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(
            split=(train_split, test_split + [len(test_single_task_benchmark.dataset)]), **default_kwargs
        )
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, test_split + [-1]), **default_kwargs)
    # It should _not_ fail with missing indices
    Benchmark(split=(train_split[:-1], test_split), **default_kwargs)


def test_multi_task_benchmark_split_verification(test_multi_task_benchmark):
    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "dataset": test_multi_task_benchmark.dataset,
        "target_cols": test_multi_task_benchmark.target_cols,
        "input_cols": test_multi_task_benchmark.input_cols,
        "metrics": test_multi_task_benchmark._metrics,
    }

    train_split = test_multi_task_benchmark._split[0]
    test_split = test_multi_task_benchmark._split[1]

    # One or more empty partitions
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split,), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, []), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, {"test": []}), **default_kwargs)
    # Non-exclusive partitions
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, test_split + train_split[:1]), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, {"test1": test_split, "test2": train_split[:1]}), **default_kwargs)
    # This should not fail (overlapping molecule, different targets)
    Benchmark(split=(train_split + [(5, [0])], test_split + [(5, [1])]), **default_kwargs)
    # Invalid indices
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split + [(-1, [0])], test_split), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split + [(0, [-1])], test_split), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, test_split + [(-1, [0])]), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(split=(train_split, test_split + [(0, [-1])]), **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(
            split=(train_split + [(len(test_multi_task_benchmark.dataset), [0])], test_split),
            **default_kwargs,
        )
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(
            split=(train_split + [(0, [len(test_multi_task_benchmark.dataset)])], test_split),
            **default_kwargs,
        )
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(
            split=(train_split, test_split + [(len(test_multi_task_benchmark.dataset), [0])]),
            **default_kwargs,
        )
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(
            split=(train_split, test_split + [(0, [len(test_multi_task_benchmark.dataset)])]),
            **default_kwargs,
        )


def test_benchmark_column_verification(test_single_task_benchmark):
    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "dataset": test_single_task_benchmark.dataset,
        "metrics": test_single_task_benchmark._metrics,
        "split": test_single_task_benchmark._split,
    }

    x_cols = test_single_task_benchmark._split[0]
    y_cols = test_single_task_benchmark._split[1]

    # Empty columns
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(input_cols=[], target_cols=y_cols, **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(input_cols=x_cols, target_cols=[], **default_kwargs)
    # Non-existent columns
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(input_cols=["invalid", "smiles"], target_cols=y_cols, **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(input_cols=x_cols, target_cols=["invalid", "expt"], **default_kwargs)


def test_benchmark_metrics_verification(test_single_task_benchmark):
    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "dataset": test_single_task_benchmark.dataset,
        "target_cols": test_single_task_benchmark.target_cols,
        "input_cols": test_single_task_benchmark.input_cols,
        "split": test_single_task_benchmark._split,
    }

    # Invalid metric
    with pytest.raises(KeyError):
        Benchmark(metrics=["invalid"], **default_kwargs)
    with pytest.raises(KeyError):
        Benchmark(metrics="invalid", **default_kwargs)
    with pytest.raises(InvalidBenchmarkError):
        Benchmark(
            metrics=test_single_task_benchmark._metrics + [test_single_task_benchmark._metrics[0]],
            **default_kwargs,
        )


def test_benchmark_split(test_single_task_benchmark):
    train, test = test_single_task_benchmark.get_train_test_split()
    assert isinstance(train, Task) and isinstance(test, Task)
    assert len(train) + len(test) <= len(test_single_task_benchmark.dataset)
