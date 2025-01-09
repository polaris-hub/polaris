import os
from time import perf_counter

import numcodecs
import numpy as np
import pandas as pd
import pytest
import zarr
from pydantic import ValidationError

from polaris.dataset import DatasetV2, Subset
from polaris.dataset._dataset_v2 import _GROUP_FORMAT_METADATA_KEY, _INDEX_ARRAY_KEY
from polaris.dataset.zarr._manifest import generate_zarr_manifest

# Helper methods


def _check_column_b_or_c_data(data, idx):
    return all(np.array_equal(data[prop], np.arange(32) + (32 * idx)) for prop in ["x", "y", "z"])


def _check_column_a_data(data, idx):
    return np.array_equal(data, np.arange(256) + (256 * idx))


# Test cases


def test_dataset_v2_get_columns(test_dataset_v2):
    assert set(test_dataset_v2.columns) == {"A", "B", "C"}


def test_dataset_v2_get_rows(test_dataset_v2):
    assert set(test_dataset_v2.rows) == set(range(100))


def test_dataset_v2_get_data(test_dataset_v2):
    indices = np.random.randint(0, len(test_dataset_v2), 5)
    for idx in indices:
        assert _check_column_a_data(test_dataset_v2.get_data(row=idx, col="A"), idx)
        assert _check_column_b_or_c_data(test_dataset_v2.get_data(row=idx, col="B"), idx)
        assert _check_column_b_or_c_data(test_dataset_v2.get_data(row=idx, col="C"), idx)


def test_dataset_v2_with_subset(test_dataset_v2):
    indices = np.random.randint(0, len(test_dataset_v2), 5)
    subset = Subset(test_dataset_v2, indices, "A", ["B", "C"])
    for i, (x, y) in enumerate(subset):
        assert _check_column_a_data(x, indices[i])
        assert _check_column_b_or_c_data(y["B"], indices[i])
        assert _check_column_b_or_c_data(y["C"], indices[i])


def test_dataset_v2_load_to_memory(test_dataset_v2):
    subset = Subset(
        dataset=test_dataset_v2,
        indices=range(100),
        input_cols=["A"],
        target_cols=["B", "C"],
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
        assert _check_column_a_data(new_dataset.get_data(row=i, col="A"), i)
        assert _check_column_b_or_c_data(new_dataset.get_data(row=i, col="B"), i)
        assert _check_column_b_or_c_data(new_dataset.get_data(row=i, col="C"), i)


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


def test_dataset_v2_with_pdbs(pdb_paths, tmp_path): ...


def test_dataset_v2_validation_index_array(zarr_archive_v2):
    root = zarr.open(zarr_archive_v2, "a")

    # Create subgroup that lacks the index array
    subgroup = root.create_group("D")
    subgroup.attrs[_GROUP_FORMAT_METADATA_KEY] = "subgroups"
    zarr.consolidate_metadata(zarr_archive_v2)

    with pytest.raises(ValidationError, match="does not have an index array"):
        DatasetV2(zarr_root_path=zarr_archive_v2)

    indices = [f"{idx}" for idx in range(100)]
    indices[-1] = "invalid"

    # Create index array that doesn't match group length (zero arrays in group, but 100 indices)
    subgroup.array(_INDEX_ARRAY_KEY, data=indices, dtype=object, object_codec=numcodecs.VLenUTF8())
    zarr.consolidate_metadata(zarr_archive_v2)

    with pytest.raises(ValidationError, match="Length of index array"):
        DatasetV2(zarr_root_path=zarr_archive_v2)

    for i in range(100):
        subgroup.create_group(str(i))
    zarr.consolidate_metadata(zarr_archive_v2)

    # Create index array that has invalid keys (last keys = 'invalid' rather than '99')
    with pytest.raises(ValidationError, match="Keys of index array"):
        DatasetV2(zarr_root_path=zarr_archive_v2)


def test_dataset_v2_validation_consistent_lengths(zarr_archive_v2):
    root = zarr.open(zarr_archive_v2, "a")

    # Change the length of one of the arrays
    root["A"].append(np.random.random((1, 256)))
    zarr.consolidate_metadata(zarr_archive_v2)

    # Subgroup has a false number of indices
    with pytest.raises(ValidationError, match="should have the same length"):
        DatasetV2(zarr_root_path=zarr_archive_v2)

    # Make the length of the different columns equal again
    subgroup = root["B"].create_group("100")
    subgroup.array("x", data=np.arange(32))
    subgroup.array("y", data=np.arange(32))
    subgroup.array("z", data=np.arange(32))

    # Directly appending a single element fails, likely because a bug in the Zarr Array API
    root["B"][_INDEX_ARRAY_KEY] = root["B"][_INDEX_ARRAY_KEY][:].tolist() + [100]

    root["C"]["x"].append(np.arange(32).reshape(1, 32))
    root["C"]["y"].append(np.arange(32).reshape(1, 32))
    root["C"]["z"].append(np.arange(32).reshape(1, 32))

    zarr.consolidate_metadata(zarr_archive_v2)
    DatasetV2(zarr_root_path=zarr_archive_v2)

    # Create subgroup with inconsistent length
    subgroup = root.create_group("D")
    subgroup.create_group("0")
    subgroup.array(_INDEX_ARRAY_KEY, data=["0"], dtype=object, object_codec=numcodecs.VLenUTF8())
    zarr.consolidate_metadata(zarr_archive_v2)

    # Subgroup has a false number of indices
    with pytest.raises(ValidationError, match="should have the same length"):
        DatasetV2(zarr_root_path=zarr_archive_v2)


def test_zarr_manifest(test_dataset_v2):
    # Assert the manifest Parquet is created
    assert test_dataset_v2.zarr_manifest_path is not None
    assert os.path.isfile(test_dataset_v2.zarr_manifest_path)

    # The root has 2 files (.zmetadata, .zgroup)
    # The A array has 1 array with 100 chunks = 100 + 1 = 101
    # The B group has 100 groups with 3 single-chunk arrays + 1 single-chunk array = 100 * (3 * 2 + 1) + 2 + 2 = 704
    # The C group has 3 arrays with 100 chunks = 3 * (100 + 1) + 2 = 305
    # Total number of files: 2 + 101 + 702 + 305 = 1112
    df = pd.read_parquet(test_dataset_v2.zarr_manifest_path)
    assert len(df) == 1112

    # Assert the manifest hash is calculated
    assert test_dataset_v2.zarr_manifest_md5sum is not None

    # Add array to Zarr archive to change the number of chunks in the dataset
    root = zarr.open(test_dataset_v2.zarr_root_path, "a")
    root.array("D", data=np.random.random((100, 256)), chunks=(1, None))

    generate_zarr_manifest(test_dataset_v2.zarr_root_path, test_dataset_v2._cache_dir)

    # Get the length of the updated manifest file
    post_change_manifest_length = len(pd.read_parquet(test_dataset_v2.zarr_manifest_path))

    # Ensure Zarr manifest has an additional 100 chunks + 1 array metadata file
    assert post_change_manifest_length == 1213


def test_dataset_v2__get_item__(test_dataset_v2):
    """Test the __getitem__() interface for the dataset V2."""

    # Get a specific cell
    assert np.array_equal(test_dataset_v2[0, "A"], np.arange(256))

    # Get a specific row
    def _check_dict_equality(d1, d2):
        assert len(d1) == len(d2)
        for k, v in d1.items():
            if isinstance(v, dict):
                _check_dict_equality(v, d2[k])
            else:
                assert np.array_equal(d1[k], d2[k])

    _check_dict_equality(
        test_dataset_v2[0],
        {
            "A": np.arange(256),
            "B": {"x": np.arange(32), "y": np.arange(32), "z": np.arange(32)},
            "C": {"x": np.arange(32), "y": np.arange(32), "z": np.arange(32)},
        },
    )
    _check_dict_equality(
        test_dataset_v2[10],
        {
            "A": np.arange(256) + 2560,
            "B": {"x": np.arange(32) + 320, "y": np.arange(32) + 320, "z": np.arange(32) + 320},
            "C": {"x": np.arange(32) + 320, "y": np.arange(32) + 320, "z": np.arange(32) + 320},
        },
    )
