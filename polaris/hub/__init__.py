from typing import List
from polaris.data import Dataset, BenchmarkSpecification


class PolarisClient:
    def __init__(self):
        pass

    def load_dataset(self, path: str) -> Dataset:
        raise NotImplementedError

    def list_datasets(self) -> List[str]:
        return []

    def load_benchmarks(self, path: str) -> BenchmarkSpecification:
        raise NotImplementedError

    def list_benchmarks(self) -> List[str]:
        return []
