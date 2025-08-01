import os
from time import perf_counter

import numcodecs
import numpy as np
import pandas as pd
import pytest
import zarr
from pydantic import ValidationError

from polaris.dataset import DatasetV2, Subset
from polaris.utils.zarr._manifest import generate_zarr_manifest


def test_dataset_v2_get_columns(test_dataset_v2):
    assert set(test_dataset_v2.columns) == {"A", "B"}


def test_dataset_v2_get_rows(test_dataset_v2):
    assert set(test_dataset_v2.rows) == set(range(100))


def test_dataset_v2_get_data(test_dataset_v2, zarr_archive):
    root = zarr.open(zarr_archive, "r")
    indices = np.random.randint(0, len(test_dataset_v2), 5)
    for idx in indices:
        assert np.array_equal(test_dataset_v2.get_data(row=idx, col="A"), root["A"][idx])
        assert np.array_equal(test_dataset_v2.get_data(row=idx, col="B"), root["B"][idx])


def test_dataset_v2_with_subset(test_dataset_v2, zarr_archive):
    root = zarr.open(zarr_archive, "r")
    indices = np.random.randint(0, len(test_dataset_v2), 5)
    subset = Subset(test_dataset_v2, indices, "A", "B")
    for i, (x, y) in enumerate(subset):
        idx = indices[i]
        assert np.array_equal(x, root["A"][idx])
        assert np.array_equal(y, root["B"][idx])


def test_dataset_v2_load_to_memory(test_dataset_v2):
    subset = Subset(
        dataset=test_dataset_v2,
        indices=range(100),
        input_cols=["A"],
        target_cols=["B"],
    )

    t1 = perf_counter()
    for x in subset:
        pass
    d1 = perf_counter() - t1

    test_dataset_v2.load_to_memory()

    t2 = perf_counter()
    for x in subset:
        pass
    d2 = perf_counter() - t2

    assert d2 < d1


def test_dataset_v2_serialization(test_dataset_v2, tmp_path):
    save_dir = str(tmp_path / "save_dir")
    path = test_dataset_v2.to_json(save_dir)
    new_dataset = DatasetV2.from_json(path)
    for i in range(5):
        assert np.array_equal(new_dataset.get_data(i, "A"), test_dataset_v2.get_data(i, "A"))
        assert np.array_equal(new_dataset.get_data(i, "B"), test_dataset_v2.get_data(i, "B"))


def test_dataset_v2_caching(test_dataset_v2, tmp_path):
    cache_dir = str(tmp_path / "cache")
    test_dataset_v2._cache_dir = cache_dir
    test_dataset_v2.cache()
    assert str(test_dataset_v2.zarr_root_path).startswith(cache_dir)


def test_dataset_v1_v2_compatibility(test_dataset, tmp_path):
    # A DataFrame is ultimately a collection of labeled numpy arrays
    # We can thus also saved these same arrays to a Zarr archive
    df = test_dataset.table

    path = str(tmp_path / "data" / "v1v2.zarr")

    root = zarr.open(path, "w")
    root.array("smiles", data=df["smiles"].values, dtype=object, object_codec=numcodecs.VLenUTF8())
    root.array("iupac", data=df["iupac"].values, dtype=object, object_codec=numcodecs.VLenUTF8())
    for col in set(df.columns) - {"smiles", "iupac"}:
        root.array(col, data=df[col].values)
    zarr.consolidate_metadata(path)

    kwargs = test_dataset.model_dump(exclude=["table", "zarr_root_path"])
    dataset = DatasetV2(**kwargs, zarr_root_path=path)

    subset_1 = Subset(dataset=test_dataset, indices=range(5), input_cols=["smiles"], target_cols=["calc"])
    subset_2 = Subset(dataset=dataset, indices=range(5), input_cols=["smiles"], target_cols=["calc"])

    for idx in range(5):
        x1, y1 = subset_1[idx]
        x2, y2 = subset_2[idx]
        assert x1 == x2
        assert y1 == y2


def test_dataset_v2_subgroup_validation(zarr_archive):
    # Create a subgroup in the Zarr archive
    root = zarr.open(zarr_archive, "a")
    subgroup = root.create_group("subgroup")
    subgroup.array("data", data=np.random.random((100, 2048)))
    zarr.consolidate_metadata(zarr_archive)

    # Creating the dataset should fail due to subgroups not being supported
    with pytest.raises(
        ValidationError,
        match="The Zarr archive of a Dataset can't have any subgroups. Found \\['subgroup'\\]",
    ):
        DatasetV2(zarr_root_path=zarr_archive)


def test_dataset_v2_validation_consistent_lengths(zarr_archive):
    root = zarr.open(zarr_archive, "a")

    # Change the length of one of the arrays
    root["A"].append(np.random.random((1, 2048)))
    zarr.consolidate_metadata(zarr_archive)

    with pytest.raises(ValidationError, match="should have the same length"):
        DatasetV2(zarr_root_path=zarr_archive)

    # Make the length of the two arrays equal again
    # shouldn't crash
    root["B"].append(np.random.random((1, 2048)))
    zarr.consolidate_metadata(zarr_archive)
    DatasetV2(zarr_root_path=zarr_archive)


def test_zarr_manifest(test_dataset_v2):
    # Assert the manifest Parquet is created
    assert test_dataset_v2.zarr_manifest_path is not None
    assert os.path.isfile(test_dataset_v2.zarr_manifest_path)

    # Assert the manifest contains 204 rows (the number "204" is chosen because
    # the Zarr archive defined in `conftest.py` contains 204 unique files)
    df = pd.read_parquet(test_dataset_v2.zarr_manifest_path)
    assert len(df) == 204

    # Assert the manifest hash is calculated
    assert test_dataset_v2.zarr_manifest_md5sum is not None

    # Add array to Zarr archive to change the number of chunks in the dataset
    root = zarr.open(test_dataset_v2.zarr_root_path, "a")
    root.array("C", data=np.random.random((100, 2048)), chunks=(1, None))

    generate_zarr_manifest(test_dataset_v2.zarr_root_path, test_dataset_v2._cache_dir)

    # Get the length of the updated manifest file
    post_change_manifest_length = len(pd.read_parquet(test_dataset_v2.zarr_manifest_path))

    # Ensure Zarr manifest has an additional 100 chunks + 1 array metadata file
    assert post_change_manifest_length == 305


def test_dataset_v2__get_item__(test_dataset_v2, zarr_archive):
    """Test the __getitem__() interface for the dataset V2."""

    # Ground truth
    root = zarr.open(zarr_archive)

    # Get a specific cell
    assert np.array_equal(test_dataset_v2[0, "A"], root["A"][0, :])

    # Get a specific row
    def _check_row_equality(d1, d2):
        assert len(d1) == len(d2)
        for k in d1:
            assert np.array_equal(d1[k], d2[k])

    _check_row_equality(test_dataset_v2[0], {"A": root["A"][0, :], "B": root["B"][0, :]})
    _check_row_equality(test_dataset_v2[10], {"A": root["A"][10, :], "B": root["B"][10, :]})
