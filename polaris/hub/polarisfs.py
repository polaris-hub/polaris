from typing import Optional
from typing import Union

import fsspec
import httpx
from polaris.utils.httpx import _log_response

from pydantic_settings import BaseSettings, SettingsConfigDict

from polaris.utils.errors import PolarisHubError
# from polaris.hub.client import PolarisHubClient

class PolarisfsSettings(BaseSettings):
    url: str = "http://localhost:3000/api/v1"

    model_config = SettingsConfigDict(env_prefix="POLARIS_")


class PolarisFSFileSystem(fsspec.AbstractFileSystem):
    sep = "/"
    protocol = "polarisfs"
    async_impl = False

    def __init__(self, polaris_client, polarisfs_url: Optional[str], default_expirations_seconds: int = 10 * 60, **kwargs):
        super().__init__(**kwargs)

        # NOTE(hadim): ugly I know xD
        if polarisfs_url is None:
            self.settings = PolarisfsSettings()
        else:
            self.settings = PolarisfsSettings(url=polarisfs_url)

        self.default_expirations_seconds = default_expirations_seconds

        self.polaris_client = polaris_client 
        self.http_client = httpx.Client(base_url=self.settings.url)
        self.http_fs = fsspec.filesystem("http")
        self.base_path = "/storage/dataset"

    def ls(self, path: str, detail=False, **kwargs):

        full_path = f"{self.base_path}/polaris/hello-world/ls/{path}"

        print("ls path: ", full_path.rstrip("/"))
        response = self.polaris_client.get(full_path.rstrip("/"), params={})
        
        response.raise_for_status()

        if not detail:
            entries = [p["name"][28:] for p in response.json()]
            print(entries)
            return entries
        else:
            detailed_entries = [{"name": p["name"][28:], "size": p["size"], "type": p["type"]} for p in response.json()]
            print(detailed_entries)
            return detailed_entries


    def cat(self, path:str, recursive=False, **kwargs):
        full_path=f"{self.base_path}/polaris/hello-world/{path}"

        response = self.polaris_client.get(full_path, params={})

        if response.status_code != 307:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise PolarisHubError("Could not get signed URL from Polaris Hub.") from error

        storage_response = self.http_client.get(response.json()["url"])

        return storage_response.json()