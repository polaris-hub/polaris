from time import perf_counter

import numpy as np
import pandas as pd
import pytest
import zarr
from datamol.utils import fs

from polaris.dataset import DatasetV1, Subset, create_dataset_from_file
from polaris.loader import load_dataset


@pytest.mark.parametrize("with_caching", [True, False])
@pytest.mark.parametrize("with_slice", [True, False])
def test_load_data(tmp_path, with_slice, with_caching):
    """Test accessing the data, specifically whether pointer columns are properly handled."""

    # Dummy data (could e.g. be a 3D structure or Image)
    arr = np.random.random((100, 100))

    tmpdir = str(tmp_path)
    zarr_path = fs.join(tmpdir, "data.zarr")

    root = zarr.open(zarr_path, "w")
    root.array("A", data=arr)
    zarr.consolidate_metadata(root.store)

    path = "A#0:5" if with_slice else "A#0"
    table = pd.DataFrame({"A": [path]}, index=[0])
    dataset = DatasetV1(table=table, annotations={"A": {"is_pointer": True}}, zarr_root_path=zarr_path)

    if with_caching:
        dataset._cache_dir = fs.join(tmpdir, "cache")
        dataset.cache()

    data = dataset.get_data(row=0, col="A")

    if with_slice:
        assert isinstance(data, tuple)
        assert len(data) == 5

        for i, d in enumerate(data):
            assert (d == arr[i]).all()

    else:
        data = dataset.get_data(row=0, col="A")
        assert (data == arr[0]).all()


def test_dataset_checksum(test_dataset):
    """Test whether the checksum is a good indicator of whether the dataset has changed in a meaningful way."""

    # Make sure the `md5sum` is part of the model dump even if not initiated yet.
    # This is important for uploads to the Hub.
    assert test_dataset._md5sum is None
    assert "md5sum" in test_dataset.model_dump()

    # Without any changes, same hash
    kwargs = test_dataset.model_dump()
    assert DatasetV1(**kwargs) == test_dataset

    # With unimportant changes, same hash
    kwargs["name"] = "changed"
    kwargs["description"] = "changed"
    kwargs["source"] = "https://changed.com"
    assert DatasetV1(**kwargs) == test_dataset

    # Check sensitivity to the row and column ordering
    kwargs["table"] = kwargs["table"].iloc[::-1]
    kwargs["table"] = kwargs["table"][kwargs["table"].columns[::-1]]
    assert DatasetV1(**kwargs) == test_dataset

    # Without any changes, but different hash
    dataset = DatasetV1(**kwargs)
    dataset._md5sum = "invalid"
    assert dataset != test_dataset

    # With changes, but same hash
    kwargs["md5sum"] = test_dataset.md5sum
    kwargs["table"] = kwargs["table"].iloc[:-1]
    assert DatasetV1(**kwargs) != test_dataset


def test_dataset_from_zarr(zarr_archive, tmpdir):
    """Test whether loading works when the zarr archive contains a single array or multiple arrays."""
    archive = zarr_archive
    dataset = create_dataset_from_file(archive, tmpdir.join("data"))

    assert len(dataset.table) == 100
    for i in range(100):
        assert dataset.get_data(row=i, col="A").shape == (2048,)
        assert dataset.get_data(row=i, col="B").shape == (2048,)


def test_dataset_from_json(test_dataset, tmpdir):
    """Test whether the dataset can be saved and loaded from json."""
    test_dataset.to_json(str(tmpdir))

    path = fs.join(str(tmpdir), "dataset.json")

    new_dataset = DatasetV1.from_json(path)
    assert test_dataset == new_dataset

    new_dataset = load_dataset(path)
    assert test_dataset == new_dataset


def test_dataset_from_zarr_to_json_and_back(zarr_archive, tmpdir):
    """
    Test whether a dataset with pointer columns, instantiated from a zarr archive,
    can be saved to and loaded from json.
    """

    json_dir = tmpdir.join("json")
    zarr_dir = tmpdir.join("zarr")

    archive = zarr_archive
    dataset = create_dataset_from_file(archive, zarr_dir)
    path = dataset.to_json(json_dir)

    new_dataset = DatasetV1.from_json(path)
    assert dataset == new_dataset

    new_dataset = load_dataset(path)
    assert dataset == new_dataset


def test_dataset_caching(zarr_archive, tmpdir):
    """Test whether the dataset remains the same after caching."""

    original_dataset = create_dataset_from_file(zarr_archive, tmpdir.join("original1"))
    cached_dataset = create_dataset_from_file(zarr_archive, tmpdir.join("original2"))
    assert original_dataset == cached_dataset

    cached_dataset._cache_dir = tmpdir.join("cached").strpath
    cache_dir = cached_dataset.cache(verify_checksum=True)
    assert cached_dataset.zarr_root_path.startswith(cache_dir)

    assert cached_dataset == original_dataset


def test_dataset_index():
    """Small test to check whether the dataset resets its index."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["X", "Y", "Z"])
    dataset = DatasetV1(table=df)
    subset = Subset(dataset=dataset, indices=[1], input_cols=["A"], target_cols=["B"])
    assert next(iter(subset)) == (np.array([2]), np.array([5]))


def test_dataset_in_memory_optimization(zarr_archive, tmpdir):
    """Check if optimization makes a default Zarr archive faster."""
    dataset = create_dataset_from_file(zarr_archive, tmpdir.join("dataset"))
    subset = Subset(dataset=dataset, indices=range(100), input_cols=["A"], target_cols=["B"])

    t1 = perf_counter()
    for x in subset:
        pass
    d1 = perf_counter() - t1

    dataset.load_to_memory()

    t2 = perf_counter()
    for x in subset:
        pass
    d2 = perf_counter() - t2

    assert d2 < d1


def test_setting_an_invalid_checksum(test_dataset):
    """Test whether setting an invalid checksum raises an error."""
    with pytest.raises(ValueError):
        test_dataset.md5sum = "invalid"


def test_checksum_verification(test_dataset):
    """Test whether setting an invalid checksum raises an error."""
    test_dataset.verify_checksum()
    test_dataset.md5sum = "0" * 32
    with pytest.raises(ValueError):
        test_dataset.verify_checksum()


def test_dataset__get_item__():
    """Test the __getitem__() interface for the dataset."""

    table = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}, index=["X", "Y", "Z"])
    dataset = DatasetV1(table=table)

    # Get a specific cell
    assert dataset["X", "A"] == 1
    assert dataset["X", "B"] == 4
    assert dataset["Y", "A"] == 2
    assert dataset["Y", "B"] == 5
    assert dataset["Z", "A"] == 3
    assert dataset["Z", "B"] == 6

    # Get a row
    assert dataset["X"] == {"A": 1, "B": 4, "C": 7}
    assert dataset["Y"] == {"A": 2, "B": 5, "C": 8}
    assert dataset["Z"] == {"A": 3, "B": 6, "C": 9}


def test_dataset__get_item__with_pointer_columns(zarr_archive, tmpdir):
    """Test the __getitem__() interface for a dataset with pointer columns (i.e. part of the data stored in Zarr)."""

    dataset = create_dataset_from_file(zarr_archive, tmpdir.join("data"))
    root = zarr.open(zarr_archive)

    # Get a specific cell
    assert np.array_equal(dataset[0, "A"], root["A"][0, :])

    # Get a specific row
    def _check_row_equality(d1, d2):
        assert len(d1) == len(d2)
        for k in d1:
            assert np.array_equal(d1[k], d2[k])

    _check_row_equality(dataset[0], {"A": root["A"][0, :], "B": root["B"][0, :]})
    _check_row_equality(dataset[10], {"A": root["A"][10, :], "B": root["B"][10, :]})
