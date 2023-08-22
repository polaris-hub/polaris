from __future__ import annotations  # Will be the default in Python >=3.11, see PEP 563

import webbrowser
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
from loguru import logger

from polaris.benchmark import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset import Dataset
from polaris.hub._client import PolarisHubClient
from polaris.utils.errors import PolarisHubError
from polaris.utils.types import HubOwner

if TYPE_CHECKING:
    # This prevents a circular import, while allowing type checking
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


def list_datasets(client_env_file: Optional[str] = None) -> list[str]:
    """List all available datasets on the Polaris Hub."""

    # TODO (cwognum): What to do with pagination, i.e. limit and offset?
    with PolarisHubClient(env_file=client_env_file) as client:
        response = client.get("/dataset")
        response.raise_for_status()
        response = response.json()
    dataset_list = [f"{HubOwner(**bm['owner'])}/{bm['name']}" for bm in response["data"]]
    return dataset_list


def get_dataset(owner: Union[str, HubOwner], name: str, client_env_file: Optional[str] = None) -> Dataset:
    """Load a dataset from the Polaris Hub."""

    with PolarisHubClient(env_file=client_env_file) as client:
        response = client.get(f"/dataset/{owner}/{name}")
        response.raise_for_status()
        response = response.json()

        storage_response = client.get(response["tableContent"]["url"])

        # This should be a 307 redirect with the signed URL
        if storage_response.status_code != 307:
            raise PolarisHubError("Could not get signed URL from Polaris Hub.")
        url = storage_response.json()["url"]

        response["table"] = client.load_from_signed_url(url, pd.read_parquet)

    return Dataset(**response)


def list_benchmarks(client_env_file: Optional[str] = None) -> list[str]:
    """List all available benchmarks on the Polaris Hub."""

    # TODO (cwognum): What to do with pagination, i.e. limit and offset?

    with PolarisHubClient(env_file=client_env_file) as client:
        response = client.get("/benchmark")
        response.raise_for_status()
        response = response.json()
    benchmarks_list = [f"{HubOwner(**bm['owner'])}/{bm['name']}" for bm in response["data"]]
    return benchmarks_list


def get_benchmark(owner: Union[str, HubOwner], name: str, client_env_file: Optional[str] = None):
    """Load an available benchmark from the Polaris Hub."""

    with PolarisHubClient(env_file=client_env_file) as client:
        response = client.get(f"/benchmark/{owner}/{name}")
        response.raise_for_status()
        response = response.json()

        print(response)

    # TODO (cwognum): Currently, the benchmark endpoints do not return the owner info for the underlying dataset.
    #  In the current version of the API, the owner is not actually used, but this will break soon.
    dataset_owner = "user_2R9qBGSZp0gANykfsNIjpx2WJib"
    response["dataset"] = get_dataset(
        dataset_owner, response["dataset"]["name"], client_env_file=client_env_file
    )

    # TODO (cwognum): As we get more complicated benchmarks, how do we still find the right subclass?
    benchmark_cls = (
        SingleTaskBenchmarkSpecification
        if len(response["targetCols"]) == 1
        else MultiTaskBenchmarkSpecification
    )
    return benchmark_cls(**response)


def upload_results_to_hub(results: BenchmarkResults, client_env_file: Optional[str] = None):
    """Upload the results to the Polaris Hub."""

    if results.benchmark_name is None or results.benchmark_owner is None:
        raise PolarisHubError("Benchmark name and owner must be set to upload results to the Polaris Hub.")

    with PolarisHubClient(env_file=client_env_file) as client:
        url = f"/benchmark/{results.benchmark_owner}/{results.benchmark_name}/result"
        response = client.post(url, json=results.model_dump(by_alias=True))
        response.raise_for_status()

    response = response.json()

    # TODO (cwognum): Use actual URL once it's available
    logger.success(
        "Your result has been successfully uploaded to the Hub. "
        f"View it here: https://polaris-hub/results/{response['id']}"
    )
    return response
