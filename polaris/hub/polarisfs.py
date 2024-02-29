import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import fsspec

from polaris.utils.errors import PolarisHubError
from polaris.utils.types import TimeoutTypes
from polaris.utils.httpx import _log_response

if TYPE_CHECKING:
    from polaris.hub.client import PolarisHubClient


class PolarisFileSystem(fsspec.AbstractFileSystem):
    """
    A file system interface for accessing datasets on the Polaris platform.

    This class extends `fsspec.AbstractFileSystem` and provides methods to list objects within a Polaris dataset
    and fetch the content of a file from the dataset.

    Note: Zarr Integration
        This file system can be used with Zarr to load multidimensional array data stored in a Dataset from
        the Polaris infrastructure. This class is needed because we otherwise cannot generate signed URLs for
        folders and Zarr is a folder based data-format.

        ```python
        fs = PolarisFileSystem(...)
        store = zarr.storage.FSStore(..., fs=polaris_fs)
        root = zarr.open(store, mode="r")
        ```

    Args:
        polaris_client: The Polaris Hub client used to make API requests.
        dataset_owner: The owner of the dataset.
        dataset_name: The name of the dataset.
    """

    sep = "/"
    protocol = "polarisfs"
    async_impl = False

    def __init__(
        self,
        polaris_client: "PolarisHubClient",
        dataset_owner: str,
        dataset_name: str,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)

        self.polaris_client = polaris_client
        self.default_timeout = self.polaris_client.settings.default_timeout

        # Prefix to remove from ls entries
        self.prefix = f"dataset/{dataset_owner}/{dataset_name}/"
        self.base_path = f"/storage/{self.prefix}"

    def ls(
        self, path: str, detail: bool = False, timeout: Optional[TimeoutTypes] = None, **kwargs: dict
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """List objects in the specified path within the Polaris dataset.

        Args:
            path: The path within the dataset to list objects.
            detail: If True, returns detailed information about each object.
            timeout: Maximum time (in seconds) to wait for the request to complete.

        Returns:
            A list of dictionaries if detail is True; otherwise, a list of object names.
        """
        if timeout is None:
            timeout = self.default_timeout

        ls_path = f"{self.base_path}ls/{path}"
        print("ls path", ls_path)

        # GET request to Polaris Hub to list objects in path
        response = self.polaris_client.get(ls_path.rstrip("/"), timeout=timeout)
        response.raise_for_status()

        if not detail:
            return [p["name"].removeprefix(self.prefix) for p in response.json()]

        return [
            {"name": p["name"].removeprefix(self.prefix), "size": p["size"], "type": p["type"]}
            for p in response.json()
        ]

    def cat_file(
        self,
        path: str,
        start: Union[int, None] = None,
        end: Union[int, None] = None,
        timeout: Optional[TimeoutTypes] = None,
        **kwargs: dict,
    ) -> bytes:
        """Fetches and returns the content of a file from the Polaris dataset.

        Args:
            path: The path to the file within the dataset.
            start: The starting index of the content to retrieve.
            end: The ending index of the content to retrieve.
            timeout: Maximum time (in seconds) to wait for the request to complete.
            kwargs: Extra arguments passed to `fsspec.open()`

        Returns:
            The content of the requested file.
        """
        if timeout is None:
            timeout = self.default_timeout

        cat_path = f"{self.base_path}{path}"
        print("cat path", cat_path)

        # GET request to Polaris Hub for signed URL of file
        response = self.polaris_client.get(cat_path)

        # This should be a 307 redirect with the signed URL
        if response.status_code != 307:
            raise PolarisHubError("Could not get signed URL from Polaris Hub.")

        signed_url = response.json()["url"]

        with fsspec.open(signed_url, "rb", **kwargs) as f:
            data = f.read()
        return data[start:end]
    
    def rm(self, path: str, recursive: bool = False, maxdepth: Optional[int] = None):

        ls_path = f"{self.base_path}ls/{path}"
        print("rm path", ls_path)

        # GET request to Polaris Hub to list objects in path
        response = self.polaris_client.get(ls_path.rstrip("/"))
        response.raise_for_status()

        return [p["name"].removeprefix(self.prefix) for p in response.json()]

    def pipe_file(self, path:str, content: Union[bytes, str], timeout: Optional[TimeoutTypes] = None, **kwargs:dict) -> None:
        """Pipes the content of a file to the Polaris dataset.

        Args:
            path: The path to the file within the dataset.
            content: The content to be piped into the file.
            timeout: Maximum time (in seconds) to wait for the request to complete.

        Returns:
            None
        """
        if timeout is None:
            timeout = self.default_timeout

        pipe_path = f"{self.base_path}put/{path}"
        print(f'pipe path: {pipe_path}')
        print(f'content: {content} \n\n')

        # PUT request to Polaris Hub to put object in path
        response = self.polaris_client.put(pipe_path, timeout=timeout, content=content)

        if response.status_code != 307:
            raise PolarisHubError("Could not get signed URL from Polaris Hub.")

        signed_url = response.json()["url"]

        headers = {"Content-Type": "application/octet-stream"}

        response = self.polaris_client.put(signed_url, content=content, headers=headers, timeout=timeout)
        
        print(_log_response(response))

        # response.raise_for_status()
        if response.status_code != 200:
            raise PolarisHubError(f"Failed to upload data to Polaris Hub. Status code: {response.status_code}")