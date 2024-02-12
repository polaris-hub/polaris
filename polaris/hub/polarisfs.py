from typing import Optional
from typing import Union

import fsspec
import httpx

from pydantic_settings import BaseSettings, SettingsConfigDict

class PolarisfsSettings(BaseSettings):
    url: str = "http://localhost:3000/api/v1"

    model_config = SettingsConfigDict(env_prefix="POLARIS_")


class PolarisFSFileSystem(fsspec.AbstractFileSystem):
    sep = "/"
    protocol = "polarisfs"
    async_impl = False

    def __init__(self, polarisfs_url: Optional[str], default_expirations_seconds: int = 10 * 60, **kwargs):
        super().__init__(**kwargs)

        # NOTE(hadim): ugly I know xD
        if polarisfs_url is None:
            self.settings = PolarisfsSettings()
        else:
            self.settings = PolarisfsSettings(url=polarisfs_url)

        self.default_expirations_seconds = default_expirations_seconds

        self.http_client = httpx.Client(base_url=self.settings.url)
        self.http_fs = fsspec.filesystem("http")

    def ls(self, polarisfs_path="storage/dataset/polaris-test/__test/ls/zarr_test_archives/1GB_many_arrays.zarr/data.zarr/", **kwargs):
        resp = self.http_client.get(polarisfs_path)
        return resp.json()['response']