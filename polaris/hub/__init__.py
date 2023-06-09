from typing import List
from polaris.dataset import Dataset, Benchmark


class PolarisClient:
    def __init__(self):
        pass

    def load_dataset(self, path: str) -> Dataset:
        raise NotImplementedError

    def list_datasets(self) -> List[str]:
        return []

    def load_benchmarks(self, path: str) -> Benchmark:
        raise NotImplementedError

    def list_benchmarks(self) -> List[str]:
        return []
