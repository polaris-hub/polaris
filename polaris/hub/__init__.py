from typing import List
from polaris.dataset import Dataset, Task


class PolarisClient:
    def __init__(self):
        pass

    def load_dataset(self, path: str) -> Dataset:
        raise NotImplementedError

    def list_datasets(self) -> List[str]:
        return []

    def load_task(self, path: str) -> Task:
        raise NotImplementedError

    def list_tasks(self) -> List[str]:
        return []
