import fsspec

from typing import Union, List, Dict, Any

from polaris.utils.errors import PolarisHubError

from httpx._types import TimeoutTypes


class PolarisFSFileSystem(fsspec.AbstractFileSystem):
    """
    A file system interface for accessing datasets on the Polaris platform.

    This class extends `fsspec.AbstractFileSystem` and provides methods to list objects within a Polaris dataset
    and fetch the content of a file from the dataset.

    Note: Zarr Integration
        This file system can be used with Zarr for working with multidimensional array data stored in the Polaris dataset.

    Args:
        polaris_client: The Polaris Hub client used to make API requests.
        dataset_owner: The owner of the dataset.
        dataset_name: The name of the dataset.
        default_expirations_seconds: Default expiration time for signed URLs.
        **kwargs: Additional keyword arguments.
    """

    sep = "/"
    protocol = "polarisfs"
    async_impl = False

    def __init__(
        self,
        polaris_client: Any,
        dataset_owner: str,
        dataset_name: str,
        default_expirations_seconds: int = 10 * 60,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)

        self.polaris_client = polaris_client

        # Prefix to remove from ls entries
        self.prefix = f"dataset/{dataset_owner}/{dataset_name}/"
        self.base_path = f"/storage/{self.prefix}"

    def ls(
        self, path: str, detail: bool = False, timeout: TimeoutTypes = (10, 200), **kwargs: dict
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """List objects in the specified path within the Polaris dataset.

        Args:
            path: The path within the dataset to list objects.
            detail: If True, returns detailed information about each object.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of dictionaries if detail is True; otherwise, a list of object names.
        """
        ls_path = f"{self.base_path}ls/{path}"

        # GET request to Polaris Hub to list objects in path
        response = self.polaris_client.get(ls_path.rstrip("/"), timeout=timeout)
        response.raise_for_status()

        if not detail:
            entries = [p["name"][len(self.prefix) :] for p in response.json()]
            return entries
        else:
            detailed_entries = [
                {"name": p["name"][len(self.prefix) :], "size": p["size"], "type": p["type"]}
                for p in response.json()
            ]
            return detailed_entries

    def cat_file(
        self, path: str, start: int = None, end: int = None, timeout: TimeoutTypes = (10, 200), **kwargs: dict
    ) -> bytes:
        """Fetches and returns the content of a file from the Polaris dataset.

        Args:
            path: The path to the file within the dataset.
            start: The starting index of the content to retrieve.
            end: The ending index of the content to retrieve.
            timeout: Maximum time (in seconds) to wait for the request to complete.
            **kwargs: Additional keyword arguments.

        Returns:
            The content of the requested file.
        """
        cat_path = f"{self.base_path}{path}"

        # GET request to Polaris Hub for signed URL of file
        response = self.polaris_client.get(cat_path)

        # This should be a 307 redirect with the signed URL
        if response.status_code != 307:
            raise PolarisHubError("Could not get signed URL from Polaris Hub.")

        storage_response = self.polaris_client.get(response.json()["url"], auth=None, timeout=timeout)
        storage_response.raise_for_status()

        return storage_response.content[start:end]
