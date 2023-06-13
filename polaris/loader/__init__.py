import fsspec
import yaml

from polaris.dataset import Dataset, BenchmarkSpecification
from polaris.hub import PolarisClient
from polaris.utils.errors import InvalidDatasetError, InvalidBenchmarkError
from polaris.utils import fs


_SUPPORTED_DATA_EXTENSIONS = ["yaml", "zarr"]
_DATASET_KWARGS_KEY = "_dataset_kwargs"


def load_dataset(path: str):
    """
    Loads the dataset. Inspired by the HF API, this can either load from a remote or local path or from the Hub.
    """

    is_file = fs.is_file(path)
    extension = fs.get_extension(path)

    if not is_file:
        # Load from the Hub
        client = PolarisClient()
        options = client.list_datasets()
        if path not in options:
            raise InvalidDatasetError(f"{path} is not a valid dataset.")
        return client.load_dataset(path)

    if extension == "zarr":
        return Dataset.from_zarr(path)
    elif extension == "yaml":
        return Dataset.from_yaml(path)

    raise NotImplementedError("This should not be reached.")


def load_benchmark(path: str):
    """
    Loads the task.

    NOTE (cwognum):
     - How to save a benchmark to a file? Should it even be a file?
     - Caching mechanism (taken from Ada?). Do we also want to use this for non-Hub files?
    """

    is_file = fs.is_file(path)
    extension = fs.get_extension(path)

    if not is_file:
        # Load from the Hub
        client = PolarisClient()
        options = client.list_benchmarks()
        if path not in options:
            raise InvalidBenchmarkError(f"{path} is not a valid task. Make sure it exists!")
        return client.load_benchmarks(path)

    with fsspec.open(path, "r") as f:
        data = yaml.safe_load(f)

    dataset_kwargs = data.pop(_DATASET_KWARGS_KEY)
    dataset = load_dataset(**dataset_kwargs)

    return BenchmarkSpecification(dataset, **data)
