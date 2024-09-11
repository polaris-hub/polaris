from typing import Any, Literal, TypeAlias

from authlib.integrations.httpx_client import OAuth2Client
from authlib.oauth2 import OAuth2Error
from authlib.oauth2.rfc6749 import OAuth2Token
from pyarrow.fs import S3FileSystem
from typing_extensions import Self

from polaris.hub.oauth import HubStorageOAuth2Token, StoragePaths, StorageTokenData
from polaris.utils.errors import PolarisHubError

Scope: TypeAlias = Literal["read", "write"]


class StorageTokenAuth:
    token: HubStorageOAuth2Token | None

    def __init__(self, token: dict[str, Any] | None, *args):
        self.token = None
        if token:
            self.set_token(token)

    def set_token(self, token: dict[str, Any] | HubStorageOAuth2Token):
        self.token = HubStorageOAuth2Token(**token) if isinstance(token, dict) else token


class StorageSession(OAuth2Client):
    """
    A context manager for managing a storage session, with token exchange and token refresh capabilities.
    Each session is associated with a specific scope and resource.
    """

    token_auth_class = StorageTokenAuth

    def __init__(self, hub_client, scope: Scope, resource: str):
        self.hub_client = hub_client
        self.resource = resource

        super().__init__(
            # OAuth2Client
            token_endpoint=hub_client.settings.hub_token_url,
            token_endpoint_auth_method="none",
            grant_type="urn:ietf:params:oauth:grant-type:token-exchange",
            scope=scope,
            # httpx.Client
            cert=hub_client.settings.ca_bundle,
        )

    def __enter__(self) -> Self:
        self.ensure_active_token()
        return self

    def _prepare_token_endpoint_body(self, body, grant_type, **kwargs):
        """
        Override to support required fields for the token exchange grant type.
        See https://datatracker.ietf.org/doc/html/rfc8693#name-request
        """
        if grant_type == "urn:ietf:params:oauth:grant-type:token-exchange":
            kwargs.update(
                {
                    "subject_token": self.hub_client.token.get("access_token"),
                    "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                    "requested_token_type": "urn:ietf:params:oauth:token-type:jwt",
                    "resource": self.resource,
                }
            )
        return super()._prepare_token_endpoint_body(body, grant_type, **kwargs)

    def fetch_token(self, **kwargs) -> dict[str, Any]:
        """
        Error handling for token fetching.
        """
        try:
            return super().fetch_token()
        except OAuth2Error as error:
            raise PolarisHubError(
                message=f"Could not obtain a token to access the storage backend. Error was: {error.error} - {error.description}"
            ) from error

    def ensure_active_token(self, token: OAuth2Token | None = None) -> bool:
        """
        Override the active check to trigger a re-fetch of the token if it is not active.
        """
        if token is None:
            token = self.token

        if token and super().ensure_active_token(token):
            return True

        # Check if external token is still valid
        if not self.hub_client.ensure_active_token():
            return False

        # If so, use it to get a new Hub token
        self.token = self.fetch_token()
        return True

    @property
    def paths(self) -> StoragePaths:
        return self.token.extra_data.paths

    @property
    def fs(self) -> S3FileSystem:
        """
        Exposes a PyArrow S3-backed file system, using the token credentials.
        Offers a higher flexibility and compatibility with other libraries expecting a file system, like Pandas.
        This might be misleading, however, as the allowed operations are limited by the token's scope.
        """
        self.ensure_active_token()
        storage_data: StorageTokenData = self.token.extra_data
        return S3FileSystem(
            access_key=storage_data.key,
            secret_key=storage_data.secret,
            session_token=f"jwt/{self.token.access_token}",
            endpoint_override=storage_data.endpoint,
            region="auto",
        )
