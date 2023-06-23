from typing import List


class PolarisClient:
    def __init__(self):
        pass

    def load_dataset(self, path: str):
        raise NotImplementedError

    def list_datasets(self) -> List[str]:
        return []

    def load_benchmark(self, path: str):
        raise NotImplementedError

    def list_benchmarks(self) -> List[str]:
        return []

    def get_active_user(self):
        return "[NOT IMPLEMENTED]"

    def upload_results(self, data):
        pass
