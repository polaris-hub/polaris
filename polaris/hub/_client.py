import json
import os
import ssl
import fsspec
import httpx

from typing import Optional, Union
from authlib.integrations.httpx_client import OAuth2Client
from authlib.oauth2.client import OAuth2Client as _OAuth2Client
from loguru import logger

import platformdirs

from polaris.hub._settings import PolarisHubSettings
from polaris.utils import fs


_CUSTOM_CA_BUNDLE_KEY = "REQUESTS_CA_BUNDLE"
_HTTPX_SSL_ERROR_CODE = "[SSL: CERTIFICATE_VERIFY_FAILED]"


class PolarisHubClient(OAuth2Client):
    """
    A client for the Polaris Hub API.

    TODO (cwognum): Should we use the Async versions of these clients instead?
        It will complicate the API due to the usage of the Python asyncio API,
        but would be worth it if we can get a performance boost.
    """

    def __init__(
        self,
        env_file: Optional[Union[str, os.PathLike]] = None,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        **kwargs,
    ):
        self._user_info = None

        if settings is None:
            settings = PolarisHubSettings(_env_file=env_file)  # type: ignore
        self.settings = settings

        # We cache the auth token by default, but allow the user to disable this.
        self.cache_auth_token = cache_auth_token
        token = kwargs.get("token")
        if fs.exists(self.auth_token_cache_path) and token is None:  # type: ignore
            with fsspec.open(self.auth_token_cache_path, "r") as fd:
                token = json.load(fd)  # type: ignore

        super().__init__(
            # OAuth2Client
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            redirect_uri=settings.callback_url,
            scope=settings.scopes,
            token_endpoint_auth_method="client_secret_post",
            token=token,
            # httpx.Client
            base_url=settings.base_url,
            verify=os.environ.get(_CUSTOM_CA_BUNDLE_KEY, True),
            # Extra
            **kwargs,
        )

    @_OAuth2Client.token.setter
    def token(self, value):
        """Override the token setter to additionally save the token to a cache"""
        if self.cache_auth_token:
            logger.info(f"Saving credentials to cache at {self.auth_token_cache_path}")
            with fsspec.open(self.auth_token_cache_path, "w") as fd:
                json.dump(value, fd)  # type: ignore
        super(OAuth2Client, type(self)).token.fset(self, value)  # type: ignore

    @property
    def auth_token_cache_path(self):
        """Where the credentials would be saved locally"""
        path = None
        if self.cache_auth_token:
            dst = platformdirs.user_cache_dir(appname="polaris")
            fs.mkdir(dst, exist_ok=True)
            path = fs.join(dst, ".polaris_auth_token.json")
        return path

    def request(self, method, url, withhold_token=False, auth=httpx.USE_CLIENT_DEFAULT, **kwargs):
        """Wraps the base request method to handle errors"""
        try:
            response = super().request(method, url, withhold_token, auth, **kwargs)
        except httpx.ConnectError as error:
            # NOTE (cwognum): In the stack-trace, the more specific SSLCertVerificationError is raised. I am however
            #  unsure how to catch one of the previous errors in the stack-trace and thus use this work-around.
            if _HTTPX_SSL_ERROR_CODE in str(error):
                raise ssl.SSLCertVerificationError(
                    "We could not verify the SSL certificate. "
                    "Please ensure the latest version of the `certifi` package is installed. "
                    "If you require the usage of a custom CA bundle, you can set the environment variable "
                    f"`{_CUSTOM_CA_BUNDLE_KEY}` to the path of your CA bundle."
                )
            raise error
        return response

    @property
    def user_info(self):
        """Use the access token to get user information."""
        if self.token is None:
            raise RuntimeError("You must be authenticated to get user information.")

        if self._user_info is None:
            user_info = self.get(
                self.settings.user_info_url,
                auth=None,  # type: ignore
                headers={"authorization": f"Bearer {self.token['access_token']}"},
            )
            user_info.raise_for_status()
            self._user_info = user_info.json()

        return self._user_info
