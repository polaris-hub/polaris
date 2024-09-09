from typing import Generator

from polaris.dataset import Dataset
from polaris.hub.client import PolarisHubClient
from polaris.utils.types import ChecksumStrategy


class DatasetHubRepository:
    """
    Repository to manage datasets in the Polaris Hub.
    """
    ENDPOINT_URL = "v1/dataset"

    def __init__(self):
        super().__init__()

    def list(self, limit: int = 100, offset: int = 0) -> Generator[Dataset, None, None]:
        with PolarisHubClient() as client:
            response = client._base_request_to_hub(
                url=self.ENDPOINT_URL, method="GET", params={
                    "limit": limit,
                    "offset": offset
                }
            )

            for dataset in response.get("data", []):
                yield Dataset(**dataset)

    def get(self, identifier: str, verify_checksum: ChecksumStrategy = "verify_unless_zarr", ) -> Dataset:
        with PolarisHubClient() as client:
            response = client._base_request_to_hub(
                url=f"{self.ENDPOINT_URL}/{identifier}", method="GET"
            )

            return Dataset(**response)

    def add(self, dataset: Dataset) -> Dataset:
        pass
