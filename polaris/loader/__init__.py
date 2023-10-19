import fsspec
import yaml

from polaris.benchmark._definitions import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset._dataset import Dataset
from polaris.hub.client import PolarisHubClient
from polaris.utils import fs
from polaris.utils.errors import InvalidBenchmarkError, InvalidDatasetError


def load_dataset(path: str):
    """
    Loads the dataset. Inspired by the HF API, this can either load from a remote or local path or from the Hub.
    """

    extension = fs.get_extension(path)
    is_file = fs.is_file(path) or extension == "zarr"

    if not is_file:
        # Load from the Hub
        client = PolarisHubClient()
        return client.get_dataset(*path.split("/"))

    if extension == "zarr":
        return Dataset.from_zarr(path)
    elif extension == "json":
        return Dataset.from_json(path)

    raise NotImplementedError("This should not be reached.")


def load_benchmark(path: str):
    """
    Loads a benchmark.
    """

    is_file = fs.is_file(path) or fs.get_extension(path) == "zarr"

    if not is_file:
        # Load from the Hub
        client = PolarisHubClient()
        return client.get_benchmark(*path.split("/"))

    with fsspec.open(path, "r") as fd:
        data = yaml.safe_load(fd)  # type: ignore

    # TODO (cwognum): As this gets more complex, how do we effectivly choose which class we should use?
    #  e.g. we might end up with a single class per benchmark.
    is_single_task = isinstance(data["target_cols"], str) or len(data["target_cols"]) == 1
    cls = SingleTaskBenchmarkSpecification if is_single_task else MultiTaskBenchmarkSpecification
    return cls.from_json(path)
