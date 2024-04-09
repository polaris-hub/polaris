import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import fsspec
from fsspec.dircache import DirCache

from polaris.utils.errors import PolarisHubError
from polaris.utils.types import TimeoutTypes

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
        use_listings_cache: bool = True,
        listings_expiry_time=None,
        max_paths=None,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)

        self.use_listings_cache = use_listings_cache
        if self.use_listings_cache:
            self.dircache = DirCache(listings_expiry_time=listings_expiry_time, max_paths=max_paths)
        else:
            self.dircache = None

        self.polaris_client = polaris_client
        self.default_timeout = self.polaris_client.settings.default_timeout

        # Prefix to remove from ls entries
        self.prefix = f"dataset/{dataset_owner}/{dataset_name}/"
        self.base_path = f"/storage/{self.prefix.rstrip('/')}"

        self.refresh = False

    @staticmethod
    def is_polarisfs_path(path: str) -> bool:
        """Check if the given path is a PolarisFS path.

        Args:
            path: The path to check.

        Returns:
            True if the path is a PolarisFS path; otherwise, False.
        """
        return path.startswith(f"{PolarisFileSystem.protocol}://")

    def ls(
        self,
        path: str,
        detail: bool = False,
        timeout: Optional[TimeoutTypes] = None,
        **kwargs: dict,
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

        ls_path = self.sep.join([self.base_path, "ls", path])

        if self.use_listings_cache and not self.refresh:
            cached_listings = self.dircache.get(path, None)
            if cached_listings is not None:
                return cached_listings if detail else [d["name"] for d in cached_listings]

            cached_listings = self.dircache.get(self._parent(path), None)
            if cached_listings is not None:
                self.dircache[path] = []
                return cached_listings if detail else [d["name"] for d in cached_listings]

        # GET request to Polaris Hub to list objects in path
        response = self.polaris_client.get(ls_path.rstrip("/"), timeout=timeout)
        response.raise_for_status()

        detailed_listings = [
            {
                "name": p["name"].removeprefix(self.prefix),
                "size": p["size"],
                "type": p["type"],
            }
            for p in response.json()
        ]

        if self.use_listings_cache:
            self.dircache[path] = detailed_listings

        if not detail:
            return [p["name"].removeprefix(self.prefix) for p in response.json()]

        return detailed_listings

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

        cat_path = self.sep.join([self.base_path, path])

        # GET request to Polaris Hub for signed URL of file
        response = self.polaris_client.get(cat_path)

        # This should be a 307 redirect with the signed URL
        if response.status_code != 307:
            raise PolarisHubError("Could not get signed URL from Polaris Hub.")

        signed_url = response.json()["url"]

        with fsspec.open(signed_url, "rb", **kwargs) as f:
            data = f.read()
        return data[start:end]

    def rm(self, path: str, recursive: bool = False, maxdepth: Optional[int] = None) -> None:
        """Remove a file or directory from the Polaris dataset.

        This method is provided for compatibility with the Zarr storage interface.
        It may be called by the Zarr store when removing a file or directory.

        Args:
            path: The path to the file or directory to be removed.
            recursive: If True, remove directories and their contents recursively.
            maxdepth: The maximum depth to recurse when removing directories.

        Returns:
            None

        Note:
            This method currently it does not perform any removal operations and is included
            as a placeholder that aligns with the Zarr interface's expectations.
        """
        raise NotImplementedError("PolarisFS does not currently support the file removal operation.")

    def pipe_file(
        self,
        path: str,
        content: Union[bytes, str],
        timeout: Optional[TimeoutTypes] = None,
        **kwargs: dict,
    ) -> None:
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

        pipe_path = self.sep.join([self.base_path, path])

        # PUT request to Polaris Hub to put object in path
        response = self.polaris_client.put(pipe_path, timeout=timeout)

        if response.status_code != 307:
            raise PolarisHubError("Could not get signed URL from Polaris Hub.")

        hub_response_body = response.json()
        signed_url = hub_response_body["url"]

        sha256_hash = hashlib.sha256(content).hexdigest()

        headers = {
            "Content-Type": "application/octet-stream",
            "x-amz-content-sha256": sha256_hash,
            "x-amz-date": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            **hub_response_body["headers"],
        }

        response = self.polaris_client.request(
            url=signed_url,
            method="PUT",
            auth=None,
            content=content,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        if self.use_listings_cache:
            new_listing = [{"name": path, "type": "file", "size": len(content)}]
            current_listings = self.dircache.get(self._parent(path), None)
            if current_listings is not None:
                new_listing.extend(current_listings)
                self.dircache[self._parent(path)] = new_listing
            else:
                self.dircache[self._parent(path)] = new_listing

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        out = self.ls(self._parent(path), detail=True, **kwargs)
        out = [o for o in out if o["name"].rstrip("/") == path]
        if out:
            return out[0]
        out = self.ls(path, detail=True, **kwargs)
        path = path.rstrip("/")
        out1 = [o for o in out if o["name"].rstrip("/") == path]
        if len(out1) == 1:
            if "size" not in out1[0]:
                out1[0]["size"] = None
            return out1[0]
        elif len(out1) > 1:
            return {"name": path, "size": 0, "type": "directory"}
        else:
            raise FileNotFoundError(path)
