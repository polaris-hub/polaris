import json

import fsspec
from datamol.utils import fs

from polaris.benchmark import MultiTaskBenchmarkSpecification, SingleTaskBenchmarkSpecification
from polaris.dataset import DatasetV1, create_dataset_from_file
from polaris.experimental._benchmark_v2 import BenchmarkV2Specification
from polaris.hub.client import PolarisHubClient
from polaris.utils.types import ChecksumStrategy


def load_dataset(path: str, verify_checksum: ChecksumStrategy = "verify_unless_zarr") -> DatasetV1:
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
        with PolarisHubClient() as client:
            client.ensure_active_token()
            return client.get_dataset(*path.split("/"), verify_checksum=verify_checksum)

    # Load from local file
    if extension == "json":
        dataset = DatasetV1.from_json(path)
    else:
        dataset = create_dataset_from_file(path)

    # Verify checksum if requested
    if dataset.should_verify_checksum(verify_checksum):
        dataset.verify_checksum()

    return dataset


def load_benchmark(path: str, verify_checksum: ChecksumStrategy = "verify_unless_zarr"):
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
        with PolarisHubClient() as client:
            client.ensure_active_token()
            return client.get_benchmark(*path.split("/"), verify_checksum=verify_checksum)

    with fsspec.open(path, "r") as fd:
        data = json.load(fd)

    is_single_task = isinstance(data["target_cols"], str) or len(data["target_cols"]) == 1

    match data["version"]:
        case 1 if is_single_task:
            cls = SingleTaskBenchmarkSpecification
        case 1:
            cls = MultiTaskBenchmarkSpecification
        case 2:
            cls = BenchmarkV2Specification
        case _:
            raise ValueError(f"Unsupported benchmark version: {data['version']}")

    benchmark = cls.from_json(path)

    # Verify checksum if requested
    if benchmark.dataset.should_verify_checksum(verify_checksum):
        benchmark.verify_checksum()

    return benchmark


def load_competition(artifact_id: str):
    """
    Loads a Polaris competition.

    On Polaris, a competition represents a secure and fair benchmark. The target labels never exist
    on the client and all results are evaluated through Polaris' servers.

    Note: Dataset is automatically loaded
        The dataset underlying the competition is automatically loaded when loading the competition.

    """
    with PolarisHubClient() as client:
        return client.get_competition(artifact_id)
