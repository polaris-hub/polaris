import fsspec
import yaml

from polaris.dataset import Dataset, SingleTaskBenchmarkSpecification, MultiTaskBenchmarkSpecification
from polaris.hub import PolarisClient
from polaris.utils.errors import InvalidDatasetError, InvalidBenchmarkError
from polaris.utils import fs


def load_dataset(path: str):
    """
    Loads the dataset. Inspired by the HF API, this can either load from a remote or local path or from the Hub.
    """

    extension = fs.get_extension(path)
    is_file = fs.is_file(path) or extension == "zarr"

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
    Loads a benchmark.
    """

    is_file = fs.is_file(path) or fs.get_extension(path) == "zarr"

    if not is_file:
        # Load from the Hub
        client = PolarisClient()
        options = client.list_benchmarks()
        if path not in options:
            raise InvalidBenchmarkError(f"{path} is not a valid task. Make sure it exists!")
        return client.load_benchmarks(path)

    with fsspec.open(path, "r") as f:
        data = yaml.safe_load(f)

    # TODO (cwognum): As this gets more complex, how do we effectivly choose which class we should use?
    #  e.g. we might end up with a single class per benchmark.
    is_single_task = isinstance(data["target_cols"], str) or len(data["target_cols"]) == 1
    cls = SingleTaskBenchmarkSpecification if is_single_task else MultiTaskBenchmarkSpecification
    return cls(**data)
