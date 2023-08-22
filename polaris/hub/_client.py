import json
import os
import ssl
from io import BytesIO
from typing import Callable, Optional, Union

import fsspec
import httpx
import platformdirs
from authlib.integrations.base_client.errors import InvalidTokenError, MissingTokenError
from authlib.integrations.httpx_client import OAuth2Client
from authlib.oauth2.client import OAuth2Client as _OAuth2Client
from httpx._types import HeaderTypes, URLTypes

from polaris.hub._settings import PolarisHubSettings
from polaris.utils import fs
from polaris.utils.errors import PolarisUnauthorizedError

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
            token=token,
            token_endpoint=self.settings.token_fetch_url,
            token_endpoint_auth_method="client_secret_post",
            # httpx.Client
            base_url=settings.base_url,
            verify=os.environ.get(_CUSTOM_CA_BUNDLE_KEY, True),
            # Extra
            **kwargs,
        )

    @_OAuth2Client.token.setter
    def token(self, value):
        """Override the token setter to additionally save the token to a cache"""
        super(OAuth2Client, type(self)).token.fset(self, value)  # type: ignore

        # We cache afterwards, because the token setter adds fields we need to save (i.e. expires_at).
        if self.cache_auth_token:
            with fsspec.open(self.auth_token_cache_path, "w") as fd:
                json.dump(value, fd)  # type: ignore

    @property
    def auth_token_cache_path(self):
        """Where the credentials would be saved locally"""
        path = None
        if self.cache_auth_token:
            dst = platformdirs.user_cache_dir(appname="polaris")
            fs.mkdir(dst, exist_ok=True)
            path = fs.join(dst, ".polaris_auth_token.json")
        return path

    @property
    def user_info(self):
        """Use the access token to get user information.

        NOTE (cwognum): We override the default `auth` and `headers` argument, since
          the defaults trigger a 530 error (Cloudflare) due to the header ordering.
          Because of this, we also have to copy some code from the base `request` method to
          make auto-refresh a token if needed. For more info, see: https://stackoverflow.com/a/62687390
        """
        if self.token is None or not self.ensure_active_token(self.token):
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
        except (MissingTokenError, InvalidTokenError, httpx.HTTPStatusError) as error:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code != 401:
                raise
            raise PolarisUnauthorizedError(
                "You are not logged in to Polaris. Use the Polaris CLI to authenticate yourself."
            ) from error
        return response

    def load_from_signed_url(self, url: URLTypes, load_fn: Callable, headers: Optional[HeaderTypes] = None):
        """Utility function to load a file from a signed URL"""
        response = self.get(url, auth=None, headers=headers)  # type: ignore
        response.raise_for_status()
        content = BytesIO(response.content)
        return load_fn(content)
