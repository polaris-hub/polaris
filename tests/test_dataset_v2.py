from copy import deepcopy
from time import perf_counter

import numcodecs
import numpy as np
import pandas as pd
import pytest
import zarr
from pydantic import ValidationError

from polaris.dataset import Subset
from polaris.dataset._factory import DatasetFactory
from polaris.dataset.converters._pdb import PDBConverter
from polaris.experimental._dataset_v2 import _INDEX_ARRAY_KEY, DatasetV2
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

    # Make changes to Zarr archive copy & rebuild the manifest
    root = zarr.open(dataset.zarr_root_path, "a")
    root["A"][0] = np.zeros(2048)
    dataset.generate_zarr_manifest()

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
    root.array("iupac", data=df["iupac"].values, dtype=object, object_codec=numcodecs.VLenUTF8())
    for col in set(df.columns) - {"smiles", "iupac"}:
        root.array(col, data=df[col].values)
    zarr.consolidate_metadata(path)

    kwargs = test_dataset.model_dump(exclude=["table", "zarr_root_path"])
    dataset = DatasetV2(**kwargs, zarr_root_path=str(path))

    subset_1 = Subset(dataset=test_dataset, indices=range(5), input_cols=["smiles"], target_cols=["calc"])
    subset_2 = Subset(dataset=dataset, indices=range(5), input_cols=["smiles"], target_cols=["calc"])

    for idx in range(5):
        x1, y1 = subset_1[idx]
        x2, y2 = subset_2[idx]
        assert x1 == x2
        assert y1 == y2


def test_dataset_v2_with_pdbs(pdb_paths, tmpdir):
    # The PDB example is interesting because it creates a more complex Zarr archive
    # that includes subgroups
    zarr_root_path = str(tmpdir.join("pdbs.zarr"))
    factory = DatasetFactory(zarr_root_path)

    # Build a V1 dataset
    converter = PDBConverter()
    factory.register_converter("pdb", converter)
    factory.add_from_files(pdb_paths, axis=0)
    dataset_v1 = factory.build()

    # Build a V2 dataset based on the V1 dataset

    # Add the magic index column to the Zarr subgroup
    root = zarr.open(zarr_root_path, "a")
    ordered_keys = [v.split("/")[-1] for v in dataset_v1.table["pdb"].values]
    root["pdb"].array(_INDEX_ARRAY_KEY, data=ordered_keys, dtype=object, object_codec=numcodecs.VLenUTF8())
    zarr.consolidate_metadata(zarr_root_path)

    # Update annotations to no longer have pointer columns
    annotations = deepcopy(dataset_v1.annotations)
    for anno in annotations.values():
        anno.is_pointer = False

    # Create the dataset
    dataset_v2 = DatasetV2(
        zarr_root_path=zarr_root_path,
        annotations=annotations,
        default_adapters=dataset_v1.default_adapters,
    )

    assert len(dataset_v1) == len(dataset_v2)
    for idx in range(len(dataset_v1)):
        pdb_1 = dataset_v1.get_data(idx, "pdb")
        pdb_2 = dataset_v2.get_data(idx, "pdb")
        assert pdb_1 == pdb_2


def test_dataset_v2_indexing(zarr_archive):
    # Create a subgroup with 100 arrays
    root = zarr.open(zarr_archive, "a")
    subgroup = root.create_group("X")
    for i in range(100):
        subgroup.array(f"{i}", data=np.array([i] * 100))

    # Index it in reverse (element 0 is the last element in the array)
    indices = [f"{idx}" for idx in range(100)][::-1]
    subgroup.array(_INDEX_ARRAY_KEY, data=indices, dtype=object, object_codec=numcodecs.VLenUTF8())
    zarr.consolidate_metadata(zarr_archive)

    # Create the dataset
    dataset = DatasetV2(zarr_root_path=zarr_archive)

    # Check that the dataset is indexed correctly
    for idx in range(100):
        assert np.array_equal(dataset.get_data(row=idx, col="X"), np.array([99 - idx] * 100))


def test_dataset_v2_validation_index_array(zarr_archive):
    root = zarr.open(zarr_archive, "a")

    # Create subgroup that lacks the index array
    subgroup = root.create_group("X")
    zarr.consolidate_metadata(zarr_archive)

    with pytest.raises(ValidationError, match="does not have an index array"):
        DatasetV2(zarr_root_path=zarr_archive)

    indices = [f"{idx}" for idx in range(100)]
    indices[-1] = "invalid"

    # Create index array that doesn't match group length (zero arrays in group, but 100 indices)
    subgroup.array(_INDEX_ARRAY_KEY, data=indices, dtype=object, object_codec=numcodecs.VLenUTF8())
    zarr.consolidate_metadata(zarr_archive)

    with pytest.raises(ValidationError, match="Length of index array"):
        DatasetV2(zarr_root_path=zarr_archive)

    for i in range(100):
        subgroup.array(f"{i}", data=np.random.random(100))
    zarr.consolidate_metadata(zarr_archive)

    # Create index array that has invalid keys (last keys = 'invalid' rather than '99')
    with pytest.raises(ValidationError, match="Keys of index array"):
        DatasetV2(zarr_root_path=zarr_archive)


def test_dataset_v2_validation_consistent_lengths(zarr_archive):
    root = zarr.open(zarr_archive, "a")

    # Change the length of one of the arrays
    root["A"].append(np.random.random((1, 2048)))
    zarr.consolidate_metadata(zarr_archive)

    # Subgroup has a false number of indices
    with pytest.raises(ValidationError, match="should have the same length"):
        DatasetV2(zarr_root_path=zarr_archive)

    # Make the length of the two arrays equal again
    # shouldn't crash
    root["B"].append(np.random.random((1, 2048)))
    zarr.consolidate_metadata(zarr_archive)
    DatasetV2(zarr_root_path=zarr_archive)

    # Create subgroup with inconsistent length
    subgroup = root.create_group("X")
    for i in range(100):
        subgroup.array(f"{i}", data=np.random.random(100))
    indices = [f"{idx}" for idx in range(100)]
    subgroup.array(_INDEX_ARRAY_KEY, data=indices, dtype=object, object_codec=numcodecs.VLenUTF8())
    zarr.consolidate_metadata(zarr_archive)

    # Subgroup has a false number of indices
    with pytest.raises(ValidationError, match="should have the same length"):
        DatasetV2(zarr_root_path=zarr_archive)


def test_zarr_manifest_created(test_dataset_v2):
    # Assert the manifest Parquet is created
    assert test_dataset_v2.zarr_manifest_path is not None

    # Assert the manifest hash is calculated
    assert test_dataset_v2.zarr_md5sum_manifest is not None

    # Assert the manifest contains 204 rows (the number "204" is chosen because
    # the Zarr archive defined in `conftest.py` contains 204 unique files)
    df = pd.read_parquet(test_dataset_v2.zarr_manifest_path)
    assert len(df) == 204
