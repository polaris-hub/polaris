from typing import Optional
from typing import Union

import fsspec
import httpx

from pydantic_settings import BaseSettings, SettingsConfigDict

from polaris.utils.errors import PolarisHubError

class PolarisfsSettings(BaseSettings):
    url: str = "http://localhost:3000/"

    model_config = SettingsConfigDict(env_prefix="POLARIS_")


class PolarisFSFileSystem(fsspec.AbstractFileSystem):
    sep = "/"
    protocol = "polarisfs"
    async_impl = False

    def __init__(self, 
                polaris_client, 
                polarisfs_url: Optional[str],
                dataset_owner: str,
                dataset_name: str,
                default_expirations_seconds: int = 10 * 60, 
                **kwargs):
        super().__init__(**kwargs)

        if polarisfs_url is None:
            self.settings = PolarisfsSettings()
        else:
            self.settings = PolarisfsSettings(url=polarisfs_url)

        self.default_expirations_seconds = default_expirations_seconds

        self.polaris_client = polaris_client 

        self.http_client = httpx.Client(base_url=self.settings.url)
        self.http_fs = fsspec.filesystem("http")

        self.base_path = f"/storage/dataset/{dataset_owner}/{dataset_name}"

        # Prefix to remove from ls entries
        self.prefix = f"dataset/{dataset_owner}/{dataset_name}/"

    def ls(self, path: str, detail=False, **kwargs):
        ls_path = f"{self.base_path}/ls/{path}"

        # GET request to Polaris Hub to list objects in path
        response = self.polaris_client.get(ls_path.rstrip("/"), params={})
        response.raise_for_status()

        if not detail:
            entries = [p["name"].replace(self.prefix, "") for p in response.json()]
            return entries
        else:
            detailed_entries = [{"name": p["name"].replace(self.prefix, ""), "size": p["size"], "type": p["type"]} for p in response.json()]
            return detailed_entries

    def cat_file(self, path:str, **kwargs):
        cat_path=f"{self.base_path}/{path}"

        # GET request to Polaris Hub for signed URL of file
        response = self.polaris_client.get(cat_path, params={})

        # This should be a 307 redirect with the signed URL
        if response.status_code != 307:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise PolarisHubError("Could not get signed URL from Polaris Hub.") from error

        storage_response = self.http_client.get(response.json()["url"], timeout=(10, 200))

        return storage_response.content