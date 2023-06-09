import zarr
import pytest
import numpy as np
import datamol as dm
import pandas as pd
from polaris.dataset import DatasetInfo, Dataset, Modality


def test_dataset_info_serialization(test_dataset_info):
    serialized = test_dataset_info.serialize()
    recovered = DatasetInfo.deserialize(serialized)
    assert recovered == test_dataset_info


def test_modality_coverage(test_data, test_dataset_info):
    assert any(c not in test_dataset_info.modalities for c in test_data.columns)
    Dataset(test_data, test_dataset_info)
    assert all(c in test_dataset_info.modalities for c in test_data.columns)


@pytest.mark.parametrize("modality", [mod for mod in list(Modality) if mod.is_pointer()])
def test_load_data(modality, tmp_path):
    # Dummy data (could e.g. be a 3D structure or Image)
    arr = np.random.random((100, 100))

    tmpdir = str(tmp_path)
    path = dm.fs.join(tmpdir, "data.zarr")
    zarr.save(path, arr)

    table = pd.DataFrame({"A": [path]}, index=[0])
    info = DatasetInfo("name", "descr", "source", {"A": modality})

    dataset = Dataset(table, info, cache_dir=tmpdir)

    # Without caching
    data = dataset.get_data(row=0, col="A")
    assert (data == arr).all()

    # With caching
    dataset.download()
    data = dataset.get_data(row=0, col="A")
    assert (data == arr).all()
