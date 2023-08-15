from __future__ import annotations  # Will be the default in Python >=3.11, see PEP 563

import webbrowser
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

from polaris.dataset import Dataset
from polaris.hub._client import PolarisHubClient

if TYPE_CHECKING:
    # This prevents a ciruclar import, while allowing type checking
    from polaris.evaluate import BenchmarkResults


def login_to_hub(
    client_env_file: Optional[str] = None,
    auto_open_browser: bool = False,
    overwrite: bool = False,
):
    """Login to the Polaris Hub using OAuth2 protocol."""

    with PolarisHubClient(env_file=client_env_file) as client:
        # Check if the user is already logged in
        if client.token is not None and not overwrite:
            info = client.user_info
            logger.info(
                f"You are already logged in to the Polaris Hub as {info['username']} ({info['email']}). "
                "Set `overwrite=True` to force re-authentication."
            )
            return

        # Step 1: Redirect user to the authorization URL
        authorization_url, _ = client.create_authorization_url(client.settings.authorize_url)

        if auto_open_browser:
            logger.info(f"Your browser has been opened to visit:\n{authorization_url}\n")
            webbrowser.open_new_tab(authorization_url)
        else:
            logger.info(f"Please visit the following URL:\n{authorization_url}\n")

        # Step 2: After user grants permission, we'll get the authorization code through the callback URL
        authorization_code = input("Please enther the authorization token: ")

        # Step 3: Exchange authorization code for an access token
        client.fetch_token(
            client.settings.token_fetch_url,
            code=authorization_code,
            grant_type="authorization_code",
        )

        logger.success(
            f"Successfully authenticated to the Polaris Hub "
            f"as `{client.user_info['username']}` ({client.user_info['email']})! 🎉"
        )


def list_datasets(client_env_file: Optional[str] = None):
    with PolarisHubClient(env_file=client_env_file) as client:
        response = client.get("/dataset")
        response.raise_for_status()
        response = response.json()
    return response["data"]


def get_dataset(owner: str, name: str, client_env_file: Optional[str] = None):
    with PolarisHubClient(env_file=client_env_file) as client:
        response = client.get(f"/dataset/{owner}/{name}")
        response.raise_for_status()
        response = response.json()
    return Dataset(**response)


def list_benchmarks(client_env_file: Optional[str] = None) -> List:
    with PolarisHubClient(env_file=client_env_file) as client:
        ...
    return []


def get_benchmark(client_env_file: Optional[str] = None):
    with PolarisHubClient(env_file=client_env_file) as client:
        ...


def upload_results_to_hub(results: BenchmarkResults, client_env_file: Optional[str] = None):
    with PolarisHubClient(env_file=client_env_file) as client:
        ...


if __name__ == "__main__":
    with PolarisHubClient(env_file="/Users/cas.wognum/polaris.env") as client:
        print(client.user_info)

    # datasets = list_datasets(client_env_file="/Users/cas.wognum/polaris.env")

    # ds = datasets[0]
    # ds["table"] = ds["tableContent"]["url"]
    # print(ds["tableContent"]["url"])
    # print(Dataset(**ds))
