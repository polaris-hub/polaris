import numpy as np
import pandas as pd
import pytest
import zarr
from pydantic import ValidationError

from polaris.dataset import Dataset
from polaris.loader import load_dataset
from polaris.utils import fs
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


def test_load_data(tmp_path):
    """Test accessing the data, specifically whether pointer columns are properly handled."""

    # Dummy data (could e.g. be a 3D structure or Image)
    arr = np.random.random((100, 100))

    tmpdir = str(tmp_path)
    path = fs.join(tmpdir, "data.zarr")
    zarr.save(path, arr)

    table = pd.DataFrame({"A": [path]}, index=[0])
    dataset = Dataset(table=table, cache_dir=tmpdir, annotations={"A": {"is_pointer": True}})

    # Without caching
    data = dataset.get_data(row=0, col="A")
    assert (data == arr).all()

    # With caching
    dataset.cache()
    data = dataset.get_data(row=0, col="A")
    assert (data == arr).all()


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


@pytest.mark.parametrize("array_per_datapoint", [True, False])
def test_dataset_from_zarr(
    test_zarr_archive_single_array, test_zarr_archive_multiple_arrays, array_per_datapoint
):
    """Test whether loading works when the zarr archive contains a single array or multiple arrays."""
    archive = test_zarr_archive_multiple_arrays if array_per_datapoint else test_zarr_archive_single_array
    dataset = Dataset.from_zarr(archive)
    assert len(dataset.table) == 100
    for i in range(100):
        assert dataset.get_data(row=i, col="A").shape == (2048,)
        assert dataset.get_data(row=i, col="B").shape == (2048,)


def test_dataset_from_zarr_equality(test_zarr_archive_single_array, test_zarr_archive_multiple_arrays):
    """
    Test whether two methods for specifying .zarr datasets lead to the same dataset.
    This specifically tests whether indexing a single arrow with our custom path syntax works.
    """
    dataset_1 = Dataset.from_zarr(test_zarr_archive_single_array)
    dataset_2 = Dataset.from_zarr(test_zarr_archive_multiple_arrays)
    assert _equality_test(dataset_1, dataset_2)


def test_dataset_from_json(test_dataset, tmpdir):
    """Test whether the dataset can be saved and loaded from json."""
    test_dataset.to_json(str(tmpdir))

    path = fs.join(str(tmpdir), "dataset.json")

    new_dataset = Dataset.from_json(path)
    assert _equality_test(test_dataset, new_dataset)

    new_dataset = load_dataset(path)
    assert _equality_test(test_dataset, new_dataset)


@pytest.mark.parametrize("array_per_datapoint", [True, False])
def test_dataset_from_zarr_to_json_and_back(
    test_zarr_archive_single_array,
    test_zarr_archive_multiple_arrays,
    array_per_datapoint,
    tmpdir,
):
    """
    Test whether a dataset with pointer columns, instantiated from a zarr archive,
    can be saved to and loaded from json.
    """

    tmpdir = str(tmpdir)
    json_dir = fs.join(tmpdir, "json")
    zarr_dir = fs.join(tmpdir, "zarr")

    archive = test_zarr_archive_multiple_arrays if array_per_datapoint else test_zarr_archive_single_array
    dataset = Dataset.from_zarr(archive)
    path = dataset.to_json(json_dir)

    new_dataset = Dataset.from_json(path)
    assert _equality_test(dataset, new_dataset)

    new_dataset = load_dataset(path)
    assert _equality_test(dataset, new_dataset)

    path = new_dataset.to_zarr(zarr_dir, "multiple" if array_per_datapoint else "single")
    new_dataset = load_dataset(path)
    assert _equality_test(dataset, new_dataset)


@pytest.mark.parametrize("array_per_datapoint", [True, False])
def test_dataset_caching(
    test_zarr_archive_single_array,
    test_zarr_archive_multiple_arrays,
    array_per_datapoint,
    tmpdir,
):
    """Test whether the dataset remains the same after caching."""
    archive = test_zarr_archive_multiple_arrays if array_per_datapoint else test_zarr_archive_single_array

    original_dataset = Dataset.from_zarr(archive)
    cached_dataset = Dataset.from_zarr(archive)
    assert original_dataset == cached_dataset

    for i in range(len(cached_dataset)):
        assert not cached_dataset.table.loc[i, "A"].startswith(original_dataset.cache_dir)
        assert not cached_dataset.table.loc[i, "B"].startswith(original_dataset.cache_dir)

    cached_dataset.cache()
    for i in range(len(cached_dataset)):
        assert cached_dataset.table.loc[i, "A"].startswith(original_dataset.cache_dir)
        assert cached_dataset.table.loc[i, "B"].startswith(original_dataset.cache_dir)

    assert _equality_test(cached_dataset, original_dataset)
