import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time
from typing import Any, Literal

from authlib.integrations.httpx_client import OAuth2Auth
from pydantic import BaseModel, Field, PositiveInt, computed_field, model_validator
from typing_extensions import Self

from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.types import AnyUrlString, HttpUrlString


class CachedTokenAuth(OAuth2Auth):
    """
    A combination of an authlib token and a httpx auth class, that will cache the token to a file.
    """

    def __init__(
        self,
        token: dict | None = None,
        token_placement="header",
        client=None,
        cache_dir=DEFAULT_CACHE_DIR,
        filename="hub_auth_token.json",
    ):
        self.token_cache_path = Path(cache_dir) / filename

        if token is None and self.token_cache_path.exists():
            token = json.loads(self.token_cache_path.read_text())

        super().__init__(token, token_placement, client)

    def set_token(self, token: dict):
        super().set_token(token)

        # Ensure the cache directory exists.
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)

        # We cache afterward, because the token setter adds fields we need to save (i.e. expires_at).
        self.token_cache_path.write_text(json.dumps(token))


class ExternalCachedTokenAuth(CachedTokenAuth):
    """
    Cached token for external authentication.
    """

    def __init__(
        self,
        token: dict | None = None,
        token_placement="header",
        client=None,
        cache_dir=DEFAULT_CACHE_DIR,
        filename="external_auth_token.json",
    ):
        super().__init__(token, token_placement, client, cache_dir, filename)


class DatasetV1Paths(BaseModel):
    root: AnyUrlString
    extension: AnyUrlString | None = None

    @computed_field
    @property
    def relative_root(self) -> str:
        return re.sub(r"^\w+://", "", self.root)

    @computed_field
    @property
    def relative_extension(self) -> str | None:
        if self.extension:
            return re.sub(r"^\w+://", "", self.extension)
        return None


class DatasetV2Paths(BaseModel):
    root: AnyUrlString
    manifest: AnyUrlString

    @computed_field
    @property
    def relative_root(self) -> str:
        return re.sub(r"^\w+://", "", self.root)

    @computed_field
    @property
    def relative_manifest(self) -> str:
        return re.sub(r"^\w+://", "", self.manifest)


class StorageTokenData(BaseModel):
    key: str
    secret: str
    endpoint: HttpUrlString
    paths: DatasetV1Paths | DatasetV2Paths = Field(union_mode="smart")


class HubOAuth2Token(BaseModel):
    """
    Model to parse and validate tokens obtained from the Polaris Hub.
    """

    issued_token_type: Literal["urn:ietf:params:oauth:token-type:jwt"] = (
        "urn:ietf:params:oauth:token-type:jwt"
    )
    token_type: Literal["Bearer"] = "Bearer"
    expires_in: PositiveInt | None = None
    expires_at: datetime | None = None
    access_token: str
    extra_data: None

    @model_validator(mode="after")
    def set_expires_at(self) -> Self:
        if self.expires_at is None and self.expires_in is not None:
            self.expires_at = datetime.fromtimestamp(time() + self.expires_in, timezone.utc)
        return self

    def is_expired(self, leeway=60) -> bool | None:
        if not self.expires_at:
            return None
        # Small timedelta to consider token as expired before it actually expires
        expiration_threshold = self.expires_at - timedelta(seconds=leeway)
        return datetime.now(timezone.utc) >= expiration_threshold

    def __getitem__(self, item) -> Any | None:
        """
        Compatibility with authlib's expectation that this is a dict
        """
        return getattr(self, item)


class HubStorageOAuth2Token(HubOAuth2Token):
    """
    Specialized model for storage tokens.
    """

    token_type: Literal["Storage"] = "Storage"
    extra_data: StorageTokenData
