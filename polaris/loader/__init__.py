import fsspec
import yaml
import pandas as pd
import datamol as dm
from typing import Optional

from polaris.dataset import Dataset, DatasetInfo, Benchmark
from polaris.hub import PolarisClient
from polaris.utils.exceptions import InvalidDatasetError, InvalidBenchmarkError


_SUPPORTED_DATA_EXTENSIONS = ["parquet"]
_SUPPORTED_METADATA_EXTENSIONS = ["yaml"]
_DATASET_KWARGS_KEY = "_dataset_kwargs"


def load_dataset(path: str, info_path: Optional[str] = None):
    """
    Loads the dataset. Inspired by the HF API, this can either load from a remote or local path or from the Hub.

    NOTE (cwognum):
     - Add caching mechanism (taken from Ada?). Do we also want to use this for non-Hub files?
     - How to handle the meta-data? Should it be a separate file? Metadata in the DataFrame?
     - How to load non-tabular data (e.g. images, conformers)?
     - Add support for the zarr format.
     - HF uses "DatasetBuilder" classes. As this logic becomes more complex, we might want to do the same.
    """

    is_file = dm.fs.is_file(path)
    extension = dm.fs.get_extension(path)

    if not is_file:
        # Load from the Hub
        client = PolarisClient()
        options = client.list_datasets()
        if path not in options:
            raise InvalidDatasetError(f"{path} is not a valid dataset.")
        return client.load_dataset(path)

    # Load from filesystem
    if extension not in _SUPPORTED_DATA_EXTENSIONS:
        raise InvalidDatasetError(
            f"{extension} is not a supported extension. Choose from {_SUPPORTED_DATA_EXTENSIONS}."
        )

    if info_path is None or dm.fs.get_extension(info_path) not in _SUPPORTED_METADATA_EXTENSIONS:
        raise ValueError(f"When loading a local dataset, `info_path` needs to be a YAML file.")

    if extension == "parquet":
        df = pd.read_parquet(path)
        with fsspec.open(info_path, "r") as f:
            data = yaml.safe_load(f)
            info = DatasetInfo.deserialize(data)
        return Dataset(df, info)

    raise NotImplementedError("This should not be reached.")


def load_benchmark(path: str):
    """
    Loads the task.

    NOTE (cwognum):
     - How to save a task to a file? Should it even be a file?
     - Caching mechanism (taken from Ada?). Do we also want to use this for non-Hub files?
    """

    is_file = dm.fs.is_file(path)
    extension = dm.fs.get_extension(path)

    if not is_file:
        # Load from the Hub
        client = PolarisClient()
        options = client.list_tasks()
        if path not in options:
            raise InvalidBenchmarkError(f"{path} is not a valid task. Make sure it exists!")
        return client.load_task(path)

    # Load from filesystem
    if extension not in _SUPPORTED_METADATA_EXTENSIONS:
        raise InvalidDatasetError(
            f"{extension} is not a supported extension. Choose from {_SUPPORTED_METADATA_EXTENSIONS}."
        )

    with fsspec.open(path, "r") as f:
        data = yaml.safe_load(f)

    dataset_kwargs = data.pop(_DATASET_KWARGS_KEY)
    dataset = load_dataset(**dataset_kwargs)

    return Benchmark(dataset, **data)
