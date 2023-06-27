import numpy as np
import pytest
import datamol as dm
import zarr

from polaris.dataset import Dataset, BenchmarkSpecification
from polaris.dataset import Modality
from polaris.utils import fs


@pytest.fixture(scope="module")
def test_data():
    return dm.data.freesolv()[:100]


@pytest.fixture(scope="module")
def test_dataset(test_data):
    return Dataset(
        table=test_data,
        name="Test",
        description="Go wild in your test cases with this awesome dataset",
        source="Datamol",
        modalities={
            "smiles": Modality.MOLECULE,
        },
    )


@pytest.fixture(scope="function")
def test_zarr_archive(tmp_path):
    tmp_path = fs.join(str(tmp_path), "data.zarr")
    root = zarr.open_group(tmp_path, mode="w")
    group_a = root.create_group("A/")
    group_b = root.create_group("B/")
    group_a.array("data", data=np.random.random((100, 2048)))
    group_b.array("data", data=np.random.random((100, 2048)))
    root.attrs["C"] = {"data": 0.0}
    root.attrs["name"] = "Test"
    root.attrs["description"] = "Go wild in your test cases with this awesome dataset"
    root.attrs["source"] = "Imagination"
    root.attrs["modalities"] = {"A": "MOLECULE_3D", "B": "IMAGE"}
    return tmp_path


@pytest.fixture(scope="module")
def test_single_task_benchmark(test_dataset):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    return BenchmarkSpecification(
        dataset=test_dataset,
        metrics=["mean_absolute_error"],
        split=(train_indices, test_indices),
        target_cols="expt",
        input_cols="smiles",
    )


@pytest.fixture(scope="module")
def test_multi_task_benchmark(test_dataset):
    # For the sake of simplicity, just use a small set of indices
    train_indices = [0, (1, [0])]
    test_indices = [(1, [1]), 2]
    return BenchmarkSpecification(
        dataset=test_dataset,
        metrics=["mean_absolute_error"],
        split=(train_indices, test_indices),
        target_cols=["expt", "calc"],
        input_cols="smiles",
    )
