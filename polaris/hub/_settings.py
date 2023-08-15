from pydantic_settings import BaseSettings, SettingsConfigDict


class PolarisHubSettings(BaseSettings):
    """Settings for the OAuth2 Polaris Hub API Client.

    These settings can be found through your Polaris Hub account.
    `client_id` and `client_secret` are user-specific and used to verify the application.
    """

    base_url: str
    authorize_url: str
    callback_url: str
    token_fetch_url: str
    user_info_url: str
    scopes: str
    client_id: str
    client_secret: str

    # Configuration of the pydantic model
    model_config = SettingsConfigDict(env_prefix="POLARIS_")
