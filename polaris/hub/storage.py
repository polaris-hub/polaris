from os import PathLike
from typing import Any, Literal, TypeAlias

import boto3
from authlib.integrations.httpx_client import OAuth2Client
from authlib.oauth2 import OAuth2Error
from authlib.oauth2.rfc6749 import OAuth2Token

from polaris.hub.oauth import HubStorageOAuth2Token, StorageTokenData
from polaris.utils.errors import PolarisHubError

Scope: TypeAlias = Literal["read", "write"]


class StorageTokenAuth:
    token: HubStorageOAuth2Token

    def __init__(self, token: dict[str, Any], token_placement='header', client=None):
        self.set_token(token)

    def set_token(self, token: dict[str, Any]):
        self.token = HubStorageOAuth2Token(**token)


class StorageSession(OAuth2Client):
    """
    A context manager for managing a storage session, with token exchange and token refresh capabilities.
    Each session is associated with a specific scope and resource.
    """
    token_auth_class = StorageTokenAuth

    def __init__(self, hub_client, scope: Scope, resource: str):
        self.hub_client = hub_client
        self.scope = scope
        self.resource = resource
        self.s3_client = None
        self.bucket: str | None = None

        super().__init__(
            # OAuth2Client
            token_endpoint=hub_client.settings.hub_token_url,
            token_endpoint_auth_method="none",
            grant_type="urn:ietf:params:oauth:grant-type:token-exchange",
            # httpx.Client
            cert=self.hub_client.ca_bundle,
        )

    def __enter__(self):
        self.ensure_active_token()
        self.init_session()
        return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     # Clean up resources if necessary
    #     pass

    def fetch_token(self, **kwargs) -> dict[str, Any]:
        try:
            return super().fetch_token(
                subject_token=self.hub_client.token,
                subject_token_type="urn:ietf:params:oauth:token-type:jwt",
                requested_token_type="urn:ietf:params:oauth:token-type:jwt",
                scope=self.scope,
                resource=self.resource,
            )
        except OAuth2Error as error:
            raise PolarisHubError(
                message=f"Could not obtain a token to access the storage backend. Error was: {error.error} - {error.description}"
            ) from error

    def ensure_active_token(self, token: OAuth2Token | None = None) -> bool:
        """
        Override the active check to trigger a re-fetch of the token if it is not active.
        """
        is_active = super().ensure_active_token(token)
        if is_active:
            return True

        # Check if external token is still valid
        if not self.hub_client.ensure_active_token():
            return False

        # If so, use it to get a new Hub token
        self.token = self.fetch_token()
        return True

    def init_session(self):
        storage_data: StorageTokenData = self.token.extra_data
        self.session = boto3.Session(
            aws_access_key_id=storage_data.key,
            aws_secret_access_key=storage_data.secret,
            aws_session_token=self.token.access_token
        )
        self.s3_client = self.session.client("s3", endpoint_url=storage_data.endpoint)

    # def check_token_validity(self):
    #     # Check if the storage token is about to expire
    #     if self.is_token_expired():
    #         self.refresh_storage_token()
    #         # Reinitialize the session with the new token
    #         self.session = boto3.Session(
    #             aws_access_key_id="your_access_key",
    #             aws_secret_access_key="your_secret_key",
    #             aws_session_token=self.storage_token,
    #         )
    #         self.s3_client = self.session.client("s3", endpoint_url="https://<your-r2-endpoint>")

    def get_resource(self, path: PathLike):
        if not self.s3_client:
            self.init_session()
        storage_data: StorageTokenData = self.token.extra_data
        paths = storage_data.paths
        self.s3_client.download_file(storage_data.bucket, paths.root, str(path))
