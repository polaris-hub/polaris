import os

import pytest

import polaris as po
from polaris.benchmark._base import BenchmarkSpecification
from polaris.dataset._base import BaseDataset


@pytest.mark.skipif(
    os.getenv("POLARIS_PASSWORD") is None and os.getenv("POLARIS_USER") is None,
    reason="This test case requires headless authentication to be set up",
)
def test_load_dataset_flow():
    dataset = po.load_dataset("polaris/hello-world")
    assert isinstance(dataset, BaseDataset)


@pytest.mark.skipif(
    os.getenv("POLARIS_PASSWORD") is None and os.getenv("POLARIS_USER") is None,
    reason="This test case requires headless authentication to be set up",
)
def test_load_benchmark_flow():
    benchmark = po.load_benchmark("polaris/hello-world-benchmark")
    assert isinstance(benchmark, BenchmarkSpecification)
