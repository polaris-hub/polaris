from urllib.parse import urljoin

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from polaris.utils.types import HttpUrlString, TimeoutTypes


class PolarisHubSettings(BaseSettings):
    """Settings for the Polaris Hub API Client.

    Attributes:
        hub_url: The URL to the main page of the Polaris Hub.
        api_url: The URL to the main entrypoint of the Polaris API.
        ca_bundle: The path to a CA bundle file for requests.
            Allows for custom SSL certificates to be used.
        default_timeout: The default timeout for requests.
        hub_token_url: The URL of the Polaris Hub token endpoint.
            A default value is generated based on the Hub URL, and this should not need to be overridden.
        api_key: The API key used for programmatic authentication to the Hub.
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
    api_key: str | None = None

    # Networking settings
    ca_bundle: str | bool | None = None
    default_timeout: TimeoutTypes = (10, 200)

    @field_validator("api_url", mode="before")
    def validate_api_url(cls, v, info: ValidationInfo):
        if v is None:
            v = urljoin(str(info.data["hub_url"]), "/api")
        return v

    @field_validator("hub_token_url", mode="before")
    def populate_hub_token_url(cls, v, info: ValidationInfo):
        if v is None:
            v = urljoin(str(info.data["hub_url"]), "/api/auth/token")
        return v
