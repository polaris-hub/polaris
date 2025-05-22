from urllib.parse import urljoin

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from polaris.utils.types import HttpUrlString, TimeoutTypes


class PolarisHubSettings(BaseSettings):
    """Settings for the OAuth2 Polaris Hub API Client.

    Info: Secrecy of these settings
        Since the Polaris Hub uses PCKE (Proof Key for Code Exchange) for OAuth2,
        these values thus do not have to be kept secret.
        See [RFC 7636](https://datatracker.ietf.org/doc/html/rfc7636) for more info.

    Attributes:
        hub_url: The URL to the main page of the Polaris Hub.
        api_url: The URL to the main entrypoint of the Polaris API.
        authorize_url: The URL of the OAuth2 authorization endpoint.
        callback_url: The URL to which the user is redirected after authorization.
        token_fetch_url: The URL of the OAuth2 token endpoint.
        user_info_url: The URL of the OAuth2 user info endpoint.
        scopes: The OAuth2 scopes that are requested.
        client_id: The OAuth2 client ID.
        ca_bundle: The path to a CA bundle file for requests.
            Allows for custom SSL certificates to be used.
        default_timeout: The default timeout for requests.
        hub_token_url: The URL of the Polaris Hub token endpoint.
            A default value is generated based on the Hub URL, and this should not need to be overridden.
        username: The username for the Polaris Hub, for the optional password-based authentication.
        password: The password for the specified username.
    """

    # Configuration of the pydantic model
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="POLARIS_", extra="ignore", env_ignore_empty=True
    )

    # Hub settings
    hub_url: HttpUrlString = "https://polarishub.io/"
    api_url: HttpUrlString | None = None
    custom_metadata_prefix: str = "X-Amz-Meta-"

    # Hub authentication settings
    hub_token_url: HttpUrlString | None = None
    username: str | None = None
    password: str | None = None

    # External authentication settings
    authorize_url: HttpUrlString = "https://clerk.polarishub.io/oauth/authorize"
    callback_url: HttpUrlString | None = None
    token_fetch_url: HttpUrlString = "https://clerk.polarishub.io/oauth/token"
    user_info_url: HttpUrlString = "https://clerk.polarishub.io/oauth/userinfo"
    scopes: str = "profile email"
    client_id: str = "agQP2xVM6JqMHvGc"

    # Networking settings
    ca_bundle: str | bool | None = None
    default_timeout: TimeoutTypes = (10, 200)

    @field_validator("api_url", mode="before")
    def validate_api_url(cls, v, info: ValidationInfo):
        if v is None:
            v = urljoin(str(info.data["hub_url"]), "/api")
        return v

    @field_validator("callback_url", mode="before")
    def validate_callback_url(cls, v, info: ValidationInfo):
        if v is None:
            v = urljoin(str(info.data["hub_url"]), "/oauth2/callback")
        return v

    @field_validator("hub_token_url", mode="before")
    def populate_hub_token_url(cls, v, info: ValidationInfo):
        if v is None:
            v = urljoin(str(info.data["hub_url"]), "/api/auth/token")
        return v
