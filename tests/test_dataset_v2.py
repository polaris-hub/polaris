from time import perf_counter

import numcodecs
import numpy as np
import pytest
import zarr

from polaris.dataset import Subset
from polaris.experimental._dataset_v2 import DatasetV2
from polaris.utils.errors import PolarisChecksumError


def test_dataset_v2_get_columns(test_dataset_v2):
    assert set(test_dataset_v2.columns) == {"A", "B"}


def test_dataset_v2_get_rows(test_dataset_v2, zarr_archive):
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


def test_dataset_v2_checksum(test_dataset_v2, tmpdir):
    # Make sure the `md5sum` is part of the model dump even if not initiated yet.
    # This is important for uploads to the Hub.
    assert test_dataset_v2._md5sum is None
    assert "md5sum" in test_dataset_v2.model_dump()

    # (1) Without any changes, same hash
    kwargs = test_dataset_v2.model_dump()
    assert DatasetV2(**kwargs) == test_dataset_v2

    # (2) With unimportant changes, same hash
    kwargs["name"] = "changed"
    kwargs["description"] = "changed"
    kwargs["source"] = "https://changed.com"
    assert DatasetV2(**kwargs) == test_dataset_v2

    # (3) Without any changes, but different hash
    dataset = DatasetV2(**kwargs)
    dataset._md5sum = "invalid"
    assert dataset != test_dataset_v2

    # (4) With changes, but same hash
    # Reset hash
    kwargs["md5sum"] = test_dataset_v2.md5sum

    # Copy Zarr data to local
    dataset = DatasetV2(**kwargs)
    save_dir = tmpdir.join("save_dir")
    dataset.to_json(save_dir, load_zarr_from_new_location=True)

    # Make changes to Zarr archive copy
    root = zarr.open(dataset.zarr_root_path, "a")
    root["A"][0] = np.zeros(2048)

    # Checksum should be different
    with pytest.raises(PolarisChecksumError):
        dataset.verify_checksum()


def test_dataset_v2_serialization(test_dataset_v2, tmpdir):
    save_dir = tmpdir.join("save_dir")
    path = test_dataset_v2.to_json(save_dir)
    new_dataset = DatasetV2.from_json(path)
    assert test_dataset_v2 == new_dataset


def test_dataset_v2_caching(test_dataset_v2, tmpdir):
    cache_dir = tmpdir.join("cache").strpath
    test_dataset_v2.cache(cache_dir, verify_checksum=True)
    assert str(test_dataset_v2.zarr_root_path).startswith(cache_dir)


def test_dataset_v1_v2_compatibility(test_dataset, tmpdir):
    # A DataFrame is ultimately a collection of labeled numpy arrays
    # We can thus also saved these same arrays to a Zarr archive
    df = test_dataset.table

    path = tmpdir.join("data/v1v2.zarr")

    root = zarr.open(path, "w")
    root.array("smiles", data=df["smiles"].values, dtype=object, object_codec=numcodecs.VLenUTF8())
    root.array("calc", data=df["calc"].values)
    zarr.consolidate_metadata(path)

    kwargs = test_dataset.model_dump(exclude=["table", "zarr_root_path"])
    dataset = DatasetV2(**kwargs, zarr_root_path=str(path))

    subset_1 = Subset(dataset=dataset, indices=range(100), input_cols=["smiles"], target_cols=["calc"])
    subset_2 = Subset(dataset=dataset, indices=range(100), input_cols=["smiles"], target_cols=["calc"])

    for idx in range(5):
        x1, y1 = subset_1[idx]
        x2, y2 = subset_2[idx]
        assert x1 == x2
        assert y1 == y2
