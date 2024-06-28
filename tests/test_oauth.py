import json

from polaris.hub.oauth import CachedTokenAuth


def test_cached_token_auth_empty_on_no_cache(tmp_path):
    filename = "test_token.json"
    auth = CachedTokenAuth(cache_dir=tmp_path, filename=filename)
    assert not auth.token


def test_cached_token_auth_reads_from_cache(tmp_path):
    filename = "test_token.json"
    cache_file = tmp_path / filename
    cache_file.write_text(
        json.dumps(
            {
                "access_token": "test_token",
                "issued_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_type": "Bearer",
                "expires_in": 576618,
                "expires_at": 1720122005,
            }
        )
    )

    auth = CachedTokenAuth(cache_dir=tmp_path, filename=filename)

    assert auth.token is not None
    assert auth.token["access_token"] == "test_token"
    assert auth.token["expires_at"] == 1720122005
    assert auth.token["expires_in"] == 576618
    assert auth.token["token_type"] == "Bearer"
    assert auth.token["issued_token_type"] == "urn:ietf:params:oauth:token-type:jwt"


def test_cached_token_auth_writes_to_cache(tmp_path):
    filename = "test_token.json"
    cache_file = tmp_path / filename

    auth = CachedTokenAuth(cache_dir=tmp_path, filename=filename)
    auth.set_token(
        {
            "access_token": "test_token",
            "issued_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_type": "Bearer",
            "expires_in": 576618,
            "expires_at": 1720122005,
        }
    )

    assert cache_file.exists()
    assert json.loads(cache_file.read_text()) == auth.token
