import pytest
from pydantic import ValidationError

from polaris import load_benchmark
from polaris.dataset import SingleTaskBenchmarkSpecification, MultiTaskBenchmarkSpecification, Subset
from polaris.utils import fs


def test_single_task_benchmark_split_verification(test_single_task_benchmark):
    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "dataset": test_single_task_benchmark.dataset,
        "target_cols": test_single_task_benchmark.target_cols,
        "input_cols": test_single_task_benchmark.input_cols,
        "metrics": test_single_task_benchmark.metrics,
    }

    train_split = test_single_task_benchmark.split[0]
    test_split = test_single_task_benchmark.split[1]

    # One or more empty partitions
    with pytest.raises(ValidationError):
        SingleTaskBenchmarkSpecification(split=(train_split,), **default_kwargs)
    with pytest.raises(ValueError):
        SingleTaskBenchmarkSpecification(split=(train_split, []), **default_kwargs)
    with pytest.raises(ValueError):
        SingleTaskBenchmarkSpecification(split=(train_split, {"test": []}), **default_kwargs)
    # Non-exclusive partitions
    with pytest.raises(ValueError):
        SingleTaskBenchmarkSpecification(split=(train_split, test_split + train_split[:1]), **default_kwargs)
    with pytest.raises(ValueError):
        SingleTaskBenchmarkSpecification(
            split=(train_split, {"test1": test_split, "test2": train_split[:1]}), **default_kwargs
        )
    # Invalid indices
    with pytest.raises(ValueError):
        SingleTaskBenchmarkSpecification(
            split=(train_split + [len(test_single_task_benchmark.dataset)], test_split), **default_kwargs
        )
    with pytest.raises(ValueError):
        SingleTaskBenchmarkSpecification(split=(train_split + [-1], test_split), **default_kwargs)
    with pytest.raises(ValueError):
        SingleTaskBenchmarkSpecification(
            split=(train_split, test_split + [len(test_single_task_benchmark.dataset)]), **default_kwargs
        )
    with pytest.raises(ValueError):
        SingleTaskBenchmarkSpecification(split=(train_split, test_split + [-1]), **default_kwargs)
    # It should _not_ fail with missing indices
    SingleTaskBenchmarkSpecification(split=(train_split[:-1], test_split), **default_kwargs)


def test_multi_task_benchmark_split_verification(test_multi_task_benchmark):
    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "dataset": test_multi_task_benchmark.dataset,
        "target_cols": test_multi_task_benchmark.target_cols,
        "input_cols": test_multi_task_benchmark.input_cols,
        "metrics": test_multi_task_benchmark.metrics,
    }

    train_split = test_multi_task_benchmark.split[0]
    test_split = test_multi_task_benchmark.split[1]

    # One or more empty partitions
    with pytest.raises(ValidationError):
        MultiTaskBenchmarkSpecification(split=(train_split,), **default_kwargs)
    with pytest.raises(ValidationError):
        MultiTaskBenchmarkSpecification(split=(train_split, []), **default_kwargs)
    with pytest.raises(ValidationError):
        MultiTaskBenchmarkSpecification(split=(train_split, {"test": []}), **default_kwargs)
    # Non-exclusive partitions
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(split=(train_split, test_split + train_split[:1]), **default_kwargs)
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(
            split=(train_split, {"test1": test_split, "test2": train_split[:1]}), **default_kwargs
        )
    # This should not fail (overlapping molecule, different targets)
    MultiTaskBenchmarkSpecification(
        split=(train_split + [(5, [0])], test_split + [(5, [1])]), **default_kwargs
    )
    # Invalid indices
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(split=(train_split + [(-1, [0])], test_split), **default_kwargs)
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(split=(train_split + [(0, [-1])], test_split), **default_kwargs)
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(split=(train_split, test_split + [(-1, [0])]), **default_kwargs)
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(split=(train_split, test_split + [(0, [-1])]), **default_kwargs)
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(
            split=(train_split + [(len(test_multi_task_benchmark.dataset), [0])], test_split),
            **default_kwargs,
        )
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(
            split=(train_split + [(0, [len(test_multi_task_benchmark.dataset)])], test_split),
            **default_kwargs,
        )
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(
            split=(train_split, test_split + [(len(test_multi_task_benchmark.dataset), [0])]),
            **default_kwargs,
        )
    with pytest.raises(ValueError):
        MultiTaskBenchmarkSpecification(
            split=(train_split, test_split + [(0, [len(test_multi_task_benchmark.dataset)])]),
            **default_kwargs,
        )


@pytest.mark.parametrize("cls", [SingleTaskBenchmarkSpecification, MultiTaskBenchmarkSpecification])
def test_benchmark_column_verification(test_single_task_benchmark, test_multi_task_benchmark, cls):
    base = (
        test_single_task_benchmark if cls == SingleTaskBenchmarkSpecification else test_multi_task_benchmark
    )

    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "dataset": base.dataset,
        "metrics": base.metrics,
        "split": base.split,
    }

    x_cols = base.input_cols
    y_cols = base.target_cols

    # Empty columns
    with pytest.raises(ValueError):
        cls(input_cols=[], target_cols=y_cols, **default_kwargs)
    with pytest.raises(ValueError):
        cls(input_cols=x_cols, target_cols=[], **default_kwargs)
    # Non-existent columns
    with pytest.raises(ValueError):
        cls(input_cols=["invalid", "smiles"], target_cols=y_cols, **default_kwargs)
    with pytest.raises(ValueError):
        cls(input_cols=x_cols, target_cols=["invalid", "expt"], **default_kwargs)


@pytest.mark.parametrize("cls", [SingleTaskBenchmarkSpecification, MultiTaskBenchmarkSpecification])
def test_benchmark_metrics_verification(test_single_task_benchmark, test_multi_task_benchmark, cls):
    # By using the fixture as a default, we know it doesn't always fail
    base = (
        test_single_task_benchmark if cls == SingleTaskBenchmarkSpecification else test_multi_task_benchmark
    )

    default_kwargs = {
        "dataset": base.dataset,
        "target_cols": base.target_cols,
        "input_cols": base.input_cols,
        "split": base.split,
    }

    # Invalid metric
    with pytest.raises(KeyError):
        cls(metrics=["invalid"], **default_kwargs)
    with pytest.raises(KeyError):
        cls(metrics="invalid", **default_kwargs)
    with pytest.raises(ValueError):
        cls(
            metrics=base.metrics + [base.metrics[0]],
            **default_kwargs,
        )


def test_benchmark_split(test_single_task_benchmark):
    train, test = test_single_task_benchmark.get_train_test_split()
    assert isinstance(train, Subset) and isinstance(test, Subset)
    assert len(train) + len(test) <= len(test_single_task_benchmark.dataset)


def test_benchmark_from_yaml(test_single_task_benchmark, tmpdir):
    test_single_task_benchmark.save(str(tmpdir))

    path = fs.join(str(tmpdir), "benchmark.yaml")
    new_benchmark = SingleTaskBenchmarkSpecification.from_yaml(path)
    assert new_benchmark == test_single_task_benchmark

    new_benchmark = load_benchmark(path)
    assert new_benchmark == test_single_task_benchmark
