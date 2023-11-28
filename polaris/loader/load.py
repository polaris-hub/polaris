import fsspec
import yaml

from polaris.benchmark._definitions import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset._dataset import Dataset
from polaris.hub.client import PolarisHubClient
from polaris.utils import fs


def load_dataset(path: str) -> Dataset:
    """
    Loads a Polaris dataset.

    In Polaris, a dataset is a tabular data structure that stores data-points in a row-wise manner.
    A dataset can have multiple modalities or targets, can be sparse
    and can be part of _one or multiple benchmarks_.

    The Polaris dataset can be loaded from the Hub or from a local or remote directory.

    - **Hub** (recommended): When loading the dataset from the Hub, you can simply
        provide the `owner/name` slug. This can be easily copied from the relevant dataset
        page on the Hub.
    - **Directory**: When loading the dataset from a directory, you should provide the path
        as returned by [`Dataset.to_json`][polaris.dataset.Dataset.to_json] or
        [`Dataset.to_zarr`][polaris.dataset.Dataset.to_zarr]. The path can be local or remote.

    Warning: Loading from `.zarr`
        Loading and saving datasets from and to `.zarr` is still experimental and currently not
        supported by the Hub.
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
    Loads a Polaris benchmark.

    In Polaris, a benchmark wraps a dataset with additional meta-data to specify the evaluation logic.

    The Polaris benchmark can be loaded from the Hub or from a local or remote directory.

    Note: Dataset is automatically loaded
        The dataset underlying the benchmark is automatically loaded when loading the benchmark.

    - **Hub** (recommended): When loading the benchmark from the Hub, you can simply
        provide the `owner/name` slug. This can be easily copied from the relevant benchmark
        page on the Hub.
    - **Directory**: When loading the benchmark from a directory, you should provide the path
        as returned by [`BenchmarkSpecification.to_json`][polaris.benchmark._base.BenchmarkSpecification.to_json].
        The path can be local or remote.
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
