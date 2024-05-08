from time import perf_counter

import numpy as np
import pandas as pd
import pytest
import zarr
from datamol.utils import fs
from pydantic import ValidationError

from polaris.dataset import Dataset, Subset, create_dataset_from_file
from polaris.loader import load_dataset
from polaris.utils.errors import PolarisChecksumError


def _equality_test(dataset_1, dataset_2):
    """
    Utility function.

    When saving a dataset to a different location, it should be considered the same dataset
    but currently the dataset checksum is used for equality and with pointer columns,
    the checksum uses the file path, not the file content (which thus changes when saving).

    See also: https://github.com/polaris-hub/polaris/issues/16
    """
    if dataset_1 == dataset_2:
        return True
    if len(dataset_1) != len(dataset_2):
        return False
    if (dataset_1.table.columns != dataset_2.table.columns).all():
        return False

    for i in range(len(dataset_1)):
        for col in dataset_1.table.columns:
            if (dataset_1.get_data(row=i, col=col) != dataset_2.get_data(row=i, col=col)).all():
                return False
    return True


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
    dataset = Dataset(table=table, annotations={"A": {"is_pointer": True}}, zarr_root_path=zarr_path)

    if with_caching:
        dataset.cache(fs.join(tmpdir, "cache"))

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

    original = test_dataset.md5sum
    assert original is not None

    # Without any changes, same hash
    kwargs = test_dataset.model_dump()
    Dataset(**kwargs)

    # With unimportant changes, same hash
    kwargs["name"] = "changed"
    kwargs["description"] = "changed"
    kwargs["source"] = "https://changed.com"
    Dataset(**kwargs)

    # Check sensitivity to the row and column ordering
    kwargs["table"] = kwargs["table"].iloc[::-1]
    kwargs["table"] = kwargs["table"][kwargs["table"].columns[::-1]]
    Dataset(**kwargs)

    def _check_for_failure(_kwargs):
        with pytest.raises(ValidationError) as error:
            Dataset(**_kwargs)
            assert error.error_count() == 1  # noqa
            assert isinstance(error.errors()[0], PolarisChecksumError)  # noqa

    # Without any changes, but different hash
    kwargs["md5sum"] = "invalid"
    _check_for_failure(kwargs)

    # With changes, but same hash
    kwargs["md5sum"] = original
    kwargs["table"] = kwargs["table"].iloc[:-1]
    _check_for_failure(kwargs)

    # With changes, but no hash
    kwargs["md5sum"] = None
    dataset = Dataset(**kwargs)
    assert dataset.md5sum is not None


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

    new_dataset = Dataset.from_json(path)
    assert _equality_test(test_dataset, new_dataset)

    new_dataset = load_dataset(path)
    assert _equality_test(test_dataset, new_dataset)


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

    new_dataset = Dataset.from_json(path)
    assert _equality_test(dataset, new_dataset)

    new_dataset = load_dataset(path)
    assert _equality_test(dataset, new_dataset)


def test_dataset_caching(zarr_archive, tmpdir):
    """Test whether the dataset remains the same after caching."""
    archive = zarr_archive

    original_dataset = create_dataset_from_file(archive, tmpdir.join("original1"))
    cached_dataset = create_dataset_from_file(archive, tmpdir.join("original2"))
    assert original_dataset == cached_dataset

    cache_dir = cached_dataset.cache(tmpdir.join("cached").strpath)
    assert cached_dataset.zarr_root_path.startswith(cache_dir)

    assert _equality_test(cached_dataset, original_dataset)


def test_dataset_index():
    """Small test to check whether the dataset resets its index."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["X", "Y", "Z"])
    dataset = Dataset(table=df)
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
