import datamol as dm
import numpy as np
import pytest
import zarr

from polaris.benchmark import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset import ColumnAnnotation, Dataset
from polaris.utils import fs
from polaris.utils.types import HubOwner, License


def _get_zarr_archive(tmp_path, datapoint_per_array: bool):
    tmp_path = fs.join(str(tmp_path), "data.zarr")
    root = zarr.open_group(tmp_path, mode="w")
    group_a = root.create_group("A/")
    group_b = root.create_group("B/")

    def _populate_group(group):
        if datapoint_per_array:
            for i in range(100):
                group.array(i, data=np.random.random((2048,)))
        else:
            group.array("data", data=np.random.random((100, 2048)))

    _populate_group(group_a)
    _populate_group(group_b)
    return tmp_path


@pytest.fixture(scope="module")
def test_data():
    data = dm.data.freesolv()[:100]
    # set an abitrary threshold for testing purpose.
    data["CLASS_expt"] = data["expt"].gt(0).astype(int).values
    data["CLASS_calc"] = data["calc"].gt(0).astype(int).values
    return data


@pytest.fixture(scope="module")
def test_org_owner():
    return HubOwner(organizationId="test-organization", slug="test-organization")


@pytest.fixture(scope="module")
def test_user_owner():
    return HubOwner(userId="test-user", slug="test-user")


@pytest.fixture(scope="module")
def test_dataset(test_data, test_org_owner):
    return Dataset(
        table=test_data,
        name="test-dataset",
        source="https://www.example.com",
        annotations={"expt": ColumnAnnotation(is_pointer=False, user_attributes={"unit": "kcal/mol"})},
        tags=["tagA", "tagB"],
        user_attributes={"attributeA": "valueA", "attributeB": "valueB"},
        owner=test_org_owner,
        license=License(id="MIT"),
    )


@pytest.fixture(scope="function")
def test_zarr_archive_multiple_arrays(tmp_path):
    return _get_zarr_archive(tmp_path, datapoint_per_array=True)


@pytest.fixture(scope="function")
def test_zarr_archive_single_array(tmp_path):
    return _get_zarr_archive(tmp_path, datapoint_per_array=False)


@pytest.fixture(scope="module")
def test_single_task_benchmark(test_dataset):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    return SingleTaskBenchmarkSpecification(
        name="single-task-benchmark",
        dataset=test_dataset,
        metrics=[
            "mean_absolute_error",
            "mean_squared_error",
            "r2",
            "spearmanr",
            "pearsonr",
            "explained_var",
        ],
        main_metric="mean_absolute_error",
        split=(train_indices, test_indices),
        target_cols="expt",
        input_cols="smiles",
    )


@pytest.fixture(scope="module")
def test_single_task_benchmark_clf(test_dataset):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    return SingleTaskBenchmarkSpecification(
        name="single-task-benchmark",
        dataset=test_dataset,
        main_metric="accuracy",
        metrics=["accuracy", "f1", "roc_auc", "pr_auc", "mcc", "cohen_kappa"],
        split=(train_indices, test_indices),
        target_cols="CLASS_expt",
        input_cols="smiles",
    )


@pytest.fixture(scope="module")
def test_single_task_benchmark_multiple_test_sets(test_dataset):
    train_indices = list(range(90))
    test_indices = {"test_1": list(range(90, 95)), "test_2": list(range(95, 100))}
    return SingleTaskBenchmarkSpecification(
        name="single-task-benchmark",
        dataset=test_dataset,
        metrics=[
            "mean_absolute_error",
            "mean_squared_error",
            "r2",
            "spearmanr",
            "pearsonr",
            "explained_var",
        ],
        main_metric="r2",
        split=(train_indices, test_indices),
        target_cols="expt",
        input_cols="smiles",
    )


@pytest.fixture(scope="module")
def test_multi_task_benchmark(test_dataset):
    # For the sake of simplicity, just use a small set of indices
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    return MultiTaskBenchmarkSpecification(
        name="multi-task-benchmark",
        dataset=test_dataset,
        main_metric="mean_absolute_error",
        metrics=[
            "mean_absolute_error",
            "mean_squared_error",
            "r2",
            "spearmanr",
            "pearsonr",
            "explained_var",
        ],
        split=(train_indices, test_indices),
        target_cols=["expt", "calc"],
        input_cols="smiles",
    )


@pytest.fixture(scope="module")
def test_multi_task_benchmark_clf(test_dataset):
    # For the sake of simplicity, just use a small set of indices
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    return MultiTaskBenchmarkSpecification(
        name="multi-task-benchmark",
        dataset=test_dataset,
        main_metric="accuracy",
        metrics=["accuracy", "f1", "roc_auc", "pr_auc", "mcc", "cohen_kappa"],
        split=(train_indices, test_indices),
        target_cols=["CLASS_expt", "CLASS_calc"],
        input_cols="smiles",
    )
