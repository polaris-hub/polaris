import polaris as po
from polaris.benchmark._base import BenchmarkSpecification
from polaris.dataset._base import BaseDataset
from polaris.hub.settings import PolarisHubSettings

settings = PolarisHubSettings()


def test_load_dataset_flow():
    dataset = po.load_dataset("polaris/hello-world")
    assert isinstance(dataset, BaseDataset)


def test_load_benchmark_flow():
    benchmark = po.load_benchmark("polaris/hello-world-benchmark")
    assert isinstance(benchmark, BenchmarkSpecification)
