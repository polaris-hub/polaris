import json
from pathlib import Path

from authlib.integrations.httpx_client import OAuth2Auth

from polaris.utils.constants import DEFAULT_CACHE_DIR


class CachedTokenAuth(OAuth2Auth):
    """
    A combination of an authlib token and a httpx auth class, that will cache the token to a file.
    """

    def __init__(
        self,
        token: dict | None = None,
        token_placement="header",
        client=None,
        cache_dir=DEFAULT_CACHE_DIR,
        filename="hub_auth_token.json",
    ):
        self.token_cache_path = Path(cache_dir) / filename

        if token is None and self.token_cache_path.exists():
            token = json.loads(self.token_cache_path.read_text())

        super().__init__(token, token_placement, client)

    def set_token(self, token: dict):
        super().set_token(token)

        # Ensure the cache directory exists.
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)

        # We cache afterward, because the token setter adds fields we need to save (i.e. expires_at).
        self.token_cache_path.write_text(json.dumps(token))


class ExternalCachedTokenAuth(CachedTokenAuth):
    """
    Cached token for external authentication.
    """

    def __init__(
        self,
        token: dict | None = None,
        token_placement="header",
        client=None,
        cache_dir=DEFAULT_CACHE_DIR,
        filename="external_auth_token.json",
    ):
        super().__init__(token, token_placement, client, cache_dir, filename)
