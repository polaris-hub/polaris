import json
from pathlib import Path

from authlib.integrations.httpx_client import OAuth2Auth

from polaris.utils.constants import DEFAULT_CACHE_DIR


class CachedTokenAuth(OAuth2Auth):
    """
    A combination of an authlib token and a httpx auth class, that will cache the token to a file.
    """

    def __init__(self, token: dict | None, token_placement='header', client=None, filename='hub_auth_token.json'):
        self.token_cache_path = Path(DEFAULT_CACHE_DIR) / filename

        if token is None and self.token_cache_path.exists():
            with open(self.token_cache_path, "r") as fd:
                token = json.load(fd)

        super().__init__(token, token_placement, client)

    def set_token(self, token: dict):
        super().set_token(token)

        # Ensure the cache directory exists.
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)

        # We cache afterward, because the token setter adds fields we need to save (i.e. expires_at).
        with open(self.token_cache_path, "w") as fd:
            json.dump(token, fd)


class ExternalCachedTokenAuth(CachedTokenAuth):
    """
    Cached token for external authentication.
    """

    def __init__(self, token: dict | None, token_placement='header', client=None):
        super().__init__(token, token_placement, client, filename='external_auth_token.json')

