import json

import fsspec
from datamol.utils import fs

from polaris.benchmark._definitions import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset import Dataset, create_dataset_from_file
from polaris.hub.client import PolarisHubClient


def load_dataset(path: str, verify_checksum: bool = True) -> Dataset:
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
        as returned by [`Dataset.to_json`][polaris.dataset.Dataset.to_json].
        The path can be local or remote.
    """

    extension = fs.get_extension(path)
    is_file = fs.is_file(path) or extension == "zarr"

    if not is_file:
        # Load from the Hub
        client = PolarisHubClient()
        return client.get_dataset(*path.split("/"), verify_checksum=verify_checksum)

    if extension == "json":
        return Dataset.from_json(path)
    return create_dataset_from_file(path)


def load_benchmark(path: str, verify_checksum: bool = True):
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
        return client.get_benchmark(*path.split("/"), verify_checksum=verify_checksum)

    with fsspec.open(path, "r") as fd:
        data = json.load(fd)

    # TODO (cwognum): As this gets more complex, how do we effectivly choose which class we should use?
    #  e.g. we might end up with a single class per benchmark.
    is_single_task = isinstance(data["target_cols"], str) or len(data["target_cols"]) == 1
    cls = SingleTaskBenchmarkSpecification if is_single_task else MultiTaskBenchmarkSpecification
    return cls.from_json(path)
