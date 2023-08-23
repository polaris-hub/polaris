from typing import Optional, Union

from pydantic import FieldValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        client_secret: The OAuth2 client secret.
        requests_ca_bundle: The path to a CA bundle file for requests.
            Allows for custom SSL certificates to be used.
    """

    hub_url: str = "https://polaris-hub.vercel.app/"
    api_url: Optional[str] = None
    authorize_url: str = "https://pure-whippet-77.clerk.accounts.dev/oauth/authorize"
    callback_url: str = "https://polaris-hub.vercel.app/oauth2/callback"
    token_fetch_url: str = "https://pure-whippet-77.clerk.accounts.dev/oauth/token"
    user_info_url: str = "https://pure-whippet-77.clerk.accounts.dev/oauth/userinfo"
    scopes: str = "profile email"
    client_id: str = "QJg8zadGwjnr6nbN"
    requests_ca_bundle: Union[str, None] = None

    # Configuration of the pydantic model
    model_config = SettingsConfigDict()

    @field_validator("api_url")
    def validate_api_url(cls, v, info: FieldValidationInfo):
        if v is None:
            v = info.data["hub_url"] + "api/v1/"
        return v
