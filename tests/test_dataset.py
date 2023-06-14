import zarr
import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from polaris.dataset import Dataset, Modality
from polaris.loader import load_dataset
from polaris.utils import fs


@pytest.mark.parametrize("modality", [mod for mod in list(Modality) if mod.is_pointer()])
def test_load_data(modality, tmp_path):
    # Dummy data (could e.g. be a 3D structure or Image)
    arr = np.random.random((100, 100))

    tmpdir = str(tmp_path)
    path = fs.join(tmpdir, "data.zarr")
    zarr.save(path, arr)

    table = pd.DataFrame({"A": [path]}, index=[0])
    dataset = Dataset(
        table=table,
        name="name",
        description="descr",
        source="source",
        modalities={"A": modality},
        cache_dir=tmpdir,
    )

    # Without caching
    data = dataset.get_data(row=0, col="A")
    assert (data == arr).all()

    # With caching
    dataset.download()
    data = dataset.get_data(row=0, col="A")
    assert (data == arr).all()


def test_dataset_checksum(test_dataset):
    original = test_dataset.checksum
    assert original is not None

    # Without any changes, same hash
    kwargs = test_dataset.dict()
    Dataset(**kwargs)

    # With unimportant changes, same hash
    kwargs["name"] = "changed"
    kwargs["description"] = "changed"
    kwargs["source"] = "changed"
    Dataset(**kwargs)

    # Check sensitivity to the row and column ordering
    kwargs["table"] = kwargs["table"].iloc[::-1]
    kwargs["table"] = kwargs["table"][kwargs["table"].columns[::-1]]
    Dataset(**kwargs)

    # Without any changes, but different hash
    kwargs["checksum"] = "invalid"
    with pytest.raises(ValidationError):
        Dataset(**kwargs)

    # With changes, but same hash
    kwargs["checksum"] = original
    kwargs["table"] = kwargs["table"].iloc[:-1]
    with pytest.raises(ValidationError):
        Dataset(**kwargs)

    # With changes, but no hash
    kwargs["checksum"] = None
    dataset = Dataset(**kwargs)
    assert dataset.checksum is not None


def test_dataset_from_zarr(test_zarr_archive):
    dataset = Dataset.from_zarr(test_zarr_archive)
    assert dataset.get_data("data", "A").shape == (100, 2048)
    assert dataset.get_data("data", "B").shape == (100, 2048)
    assert dataset.get_data("data", "C") == 0.0


def test_dataset_from_yaml(test_dataset, tmpdir):
    test_dataset.save(str(tmpdir))

    path = fs.join(str(tmpdir), "dataset.yaml")
    new_dataset = Dataset.from_yaml(path)
    assert test_dataset == new_dataset

    new_dataset = load_dataset(path)
    assert test_dataset == new_dataset
