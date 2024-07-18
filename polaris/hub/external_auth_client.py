import webbrowser
from typing import Optional

from authlib.common.security import generate_token
from authlib.integrations.base_client import OAuthError
from authlib.integrations.httpx_client import OAuth2Client
from authlib.oauth2 import TokenAuth
from loguru import logger

from polaris.hub.oauth import ExternalCachedTokenAuth
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.errors import PolarisUnauthorizedError


class ExternalAuthClient(OAuth2Client):
    """
    This authentication client is used to obtain OAuth 2 tokens from Polaris's external OAuth2 server.
    These can in turn be used to obtain Polaris Hub tokens.

    Note: Internal use
        This class is intended for internal use by the `PolarisHubClient` class, and you should not have to
        interact with it directly.
    """

    def __init__(
        self,
        settings: PolarisHubSettings,
        cache_auth_token: bool = True,
        **kwargs: dict,
    ):
        """
        Args:
            settings: A `PolarisHubSettings` instance.
            cache_auth_token: Whether to cache the auth token to a file.
            **kwargs: Additional keyword arguments passed to the authlib `OAuth2Client` constructor.
        """
        self._user_info = None

        # We cache the auth token by default, but allow the user to disable this.
        self.token_auth_class = ExternalCachedTokenAuth if cache_auth_token else TokenAuth

        self.settings = settings

        self.code_verifier = generate_token(48)

        super().__init__(
            # OAuth2Client
            client_id=self.settings.client_id,
            redirect_uri=self.settings.callback_url,
            scope=self.settings.scopes,
            token_endpoint=self.settings.token_fetch_url,
            code_challenge_method="S256",
            # httpx.Client
            timeout=self.settings.default_timeout,
            cert=self.settings.ca_bundle,
            # Extra
            **kwargs,
        )

    def create_authorization_url(self, **kwargs) -> tuple[str, Optional[str]]:
        """Light wrapper to automatically pass in the right URL."""
        return super().create_authorization_url(
            url=self.settings.authorize_url, code_verifier=self.code_verifier, **kwargs
        )

    def fetch_token(self, **kwargs) -> dict:
        """Light wrapper to automatically pass in the right URL"""
        return super().fetch_token(
            url=self.settings.token_fetch_url, code_verifier=self.code_verifier, **kwargs
        )

    def ensure_active_token(self, token) -> bool:
        try:
            return super().ensure_active_token(token) or False
        except OAuthError:
            # The refresh attempt can fail with this error
            return False

    @property
    def user_info(self) -> dict:
        """
        Get information about the currently logged-in user through the OAuth2 User Info flow."""

        # NOTE (cwognum): We override the default `auth` and `headers` argument, since
        #  the defaults trigger a 530 error (Cloudflare) due to the header ordering.
        #  Because of this, we also have to copy some code from the base `request` method to
        #  make auto-refresh a token if needed. For more info, see: https://stackoverflow.com/a/62687390

        try:
            if self.token is None or not self.ensure_active_token(self.token):
                raise PolarisUnauthorizedError
        except OAuthError:
            raise PolarisUnauthorizedError

        if self._user_info is None:
            user_info = self.get(
                self.settings.user_info_url,
                auth=None,  # type: ignore
                headers={"authorization": f"Bearer {self.token['access_token']}"},
            )
            user_info.raise_for_status()
            self._user_info = user_info.json()

        return self._user_info

    def interactive_login(self, overwrite: bool = False, auto_open_browser: bool = True):
        """
        Login to the Polaris Hub using an interactive flow, through a Web browser.

        Warning: Headless authentication
            It is currently not possible to log in to the Polaris Hub without a browser.
            See [this GitHub issue](https://github.com/polaris-hub/polaris/issues/30) for more info.

        Args:
            overwrite: Whether to overwrite the current token if the user is already logged in.
            auto_open_browser: Whether to automatically open the browser to visit the authorization URL.
        """

        # Check if the user is already logged in
        token_is_valid = self.token is not None and self.ensure_active_token(self.token)
        if token_is_valid and not overwrite:
            try:
                info = self.user_info
                logger.info(
                    f"You are already logged in to the Polaris Hub as {info['email']}. Set `overwrite=True` to force re-authentication."
                )
                return
            except PolarisUnauthorizedError:
                pass

        # Step 1: Redirect user to the authorization URL
        authorization_url, _ = self.create_authorization_url()

        if auto_open_browser:
            logger.info(f"Your browser has been opened to visit:\n{authorization_url}\n")
            webbrowser.open_new_tab(authorization_url)
        else:
            logger.info(f"Please visit the following URL:\n{authorization_url}\n")

        # Step 2: After user grants permission, we'll get the authorization code through the callback URL
        authorization_code = input("Please enter the authorization token: ")

        # Step 3: Exchange authorization code for an access token
        self.fetch_token(code=authorization_code, grant_type="authorization_code")

        logger.success(f"Successfully authenticated to the Polaris Hub as `{self.user_info['email']}`! 🎉")
