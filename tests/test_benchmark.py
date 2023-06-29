import pytest
from pydantic import ValidationError

from polaris import load_benchmark
from polaris.dataset import SingleTaskBenchmarkSpecification, MultiTaskBenchmarkSpecification, Subset
from polaris.utils import fs
from polaris.utils.errors import PolarisChecksumError


@pytest.mark.parametrize("is_single_task", [True, False])
def test_split_verification(is_single_task, test_single_task_benchmark, test_multi_task_benchmark):
    obj = test_single_task_benchmark if is_single_task else test_multi_task_benchmark
    cls = SingleTaskBenchmarkSpecification if is_single_task else MultiTaskBenchmarkSpecification

    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "dataset": obj.dataset,
        "target_cols": obj.target_cols,
        "input_cols": obj.input_cols,
        "metrics": obj.metrics,
    }

    train_split = obj.split[0]
    test_split = obj.split[1]

    # One or more empty partitions
    with pytest.raises(ValidationError):
        cls(split=(train_split,), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, []), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, {"test": []}), **default_kwargs)
    # Non-exclusive partitions
    with pytest.raises(ValidationError):
        cls(split=(train_split, test_split + train_split[:1]), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, {"test1": test_split, "test2": train_split[:1]}), **default_kwargs)
    # Invalid indices
    with pytest.raises(ValidationError):
        cls(split=(train_split + [len(obj.dataset)], test_split), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split + [-1], test_split), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, test_split + [len(obj.dataset)]), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, test_split + [-1]), **default_kwargs)
    # Duplicate indices
    with pytest.raises(ValidationError):
        cls(split=(train_split + train_split[:1], test_split), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, test_split + test_split[:1]), **default_kwargs)
    # It should _not_ fail with missing indices
    cls(split=(train_split[:-1], test_split), **default_kwargs)


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
    test_single_task_benchmark.to_yaml(str(tmpdir))

    path = fs.join(str(tmpdir), "benchmark.yaml")
    new_benchmark = SingleTaskBenchmarkSpecification.from_yaml(path)
    assert new_benchmark == test_single_task_benchmark

    new_benchmark = load_benchmark(path)
    assert new_benchmark == test_single_task_benchmark


@pytest.mark.parametrize("is_single_task", [True, False])
def test_benchmark_checksum(is_single_task, test_single_task_benchmark, test_multi_task_benchmark):
    obj = test_single_task_benchmark if is_single_task else test_multi_task_benchmark
    cls = SingleTaskBenchmarkSpecification if is_single_task else MultiTaskBenchmarkSpecification

    original = obj.md5sum
    assert original is not None

    # --- Test that the checksum is the same with insignificant changes ---

    # Without any changes, same hash
    kwargs = obj.dict()
    cls(**kwargs)

    # With a different ordering of the target columns
    kwargs["target_cols"] = kwargs["target_cols"][::-1]
    cls(**kwargs)

    # With a different ordering of the metrics
    kwargs["metrics"] = kwargs["metrics"][::-1]
    cls(**kwargs)

    # With a different ordering of the split
    kwargs["split"] = kwargs["split"][::-1]
    cls(**kwargs)

    # --- Test that the checksum is NOT the same ---
    def _check_for_failure(_kwargs):
        with pytest.raises(ValidationError) as error:
            cls(**_kwargs)
            assert error.error_count() == 1  # noqa
            assert isinstance(error.errors()[0], PolarisChecksumError)  # noqa

    # Split
    kwargs = obj.dict()
    kwargs["split"] = kwargs["split"][0][1:] + [-1], kwargs["split"][1]
    _check_for_failure(kwargs)

    kwargs = obj.dict()
    kwargs["split"] = kwargs["split"][0], kwargs["split"][1][1:] + [-1]
    _check_for_failure(kwargs)

    # Metrics
    kwargs = obj.dict()
    kwargs["metrics"] = kwargs["metrics"][1:] + ["accuracy"]
    _check_for_failure(kwargs)

    # Target columns
    kwargs = obj.dict()
    kwargs["target_cols"] = kwargs["target_cols"][1:] + ["iupac"]
    _check_for_failure(kwargs)

    kwargs = obj.dict()
    kwargs["input_cols"] = kwargs["input_cols"][1:] + ["iupac"]
    _check_for_failure(kwargs)

    # --- Don't fail if not checksum is provided ---
    kwargs["md5sum"] = None
    dataset = cls(**kwargs)
    assert dataset.md5sum is not None