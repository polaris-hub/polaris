import pytest
import datamol as dm

from polaris.dataset import DatasetInfo, Dataset, Benchmark
from polaris.dataset import Modality


@pytest.fixture(scope="module")
def test_data():
    return dm.data.freesolv()[:100]


@pytest.fixture(scope="module")
def test_dataset_info():
    return DatasetInfo(
        name="Test",
        description="Go wild in your test cases with this awesome dataset",
        source="Datamol",
        modalities={
            "expt": Modality.TARGET,
            "calc": Modality.TARGET,
            "smiles": Modality.MOLECULE,
        },
    )


@pytest.fixture(scope="module")
def test_dataset(test_data, test_dataset_info):
    return Dataset(test_data, test_dataset_info)


@pytest.fixture(scope="module")
def test_single_task(test_dataset):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    return Benchmark(
        dataset=test_dataset,
        metrics=["mean_absolute_error"],
        split=(train_indices, test_indices),
        target_cols="expt",
        input_cols="smiles",
    )


@pytest.fixture(scope="module")
def test_multi_task(test_dataset):
    # For the sake of simplicity, just use a small set of indices
    train_indices = [(0, [0, 1]), (1, [0])]
    test_indices = [(1, [1]), (2, [0, 1])]
    return Benchmark(
        dataset=test_dataset,
        metrics=["mean_absolute_error"],
        split=(train_indices, test_indices),
        target_cols=["expt", "calc"],
        input_cols="smiles",
    )
