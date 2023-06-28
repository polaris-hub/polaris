from typing import List, Optional

from polaris.utils.errors import PolarisUnauthorizedError


class PolarisClient:
    """Singleton class to interact with the Polaris Hub."""

    _instance = None

    @staticmethod
    def get_client():
        if PolarisClient._instance is None:
            PolarisClient()
        return PolarisClient._instance

    def __init__(self):
        if PolarisClient._instance is not None:
            raise RuntimeError(
                f"{self.__class__.__name__} is a singleton and should not be instantiated directly. "
                f"Use PolarisClient.get_client() instead."
            )
        PolarisClient._instance = self

    # ===================- Methods that do not require logging in -=================== #
    def load_dataset(self, path: str):
        raise NotImplementedError

    def list_datasets(self) -> List[str]:
        return []

    def load_benchmark(self, path: str):
        raise NotImplementedError

    def list_benchmarks(self) -> List[str]:
        return []

    def get_active_user(self) -> Optional[str]:
        return None

    # ====================- Methods that do require logging in -==================== #
    def upload_results_to_hub(self, data):
        if self.get_active_user() is None:
            raise PolarisUnauthorizedError("You must be logged in to upload results to the Hub.")
        raise NotImplementedError
