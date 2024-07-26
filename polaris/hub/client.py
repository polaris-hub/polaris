import json
import ssl
from hashlib import md5
from io import BytesIO
from typing import Callable, get_args
from urllib.parse import urljoin

import certifi
import httpx
import pandas as pd
import zarr
from authlib.integrations.base_client.errors import InvalidTokenError, MissingTokenError
from authlib.integrations.httpx_client import OAuth2Client, OAuthError
from authlib.oauth2 import TokenAuth
from authlib.oauth2.rfc6749 import OAuth2Token
from httpx import HTTPStatusError, Response
from httpx._types import HeaderTypes, URLTypes
from loguru import logger

from polaris.benchmark import (
    BenchmarkSpecification,
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset import Dataset
from polaris.evaluate import BenchmarkResults
from polaris.hub.external_auth_client import ExternalAuthClient
from polaris.hub.oauth import CachedTokenAuth
from polaris.hub.polarisfs import PolarisFileSystem
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.context import ProgressIndicator, tmp_attribute_change
from polaris.utils.errors import (
    InvalidDatasetError,
    PolarisCreateArtifactError,
    PolarisHubError,
    PolarisRetrieveArtifactError,
    PolarisUnauthorizedError,
)
from polaris.utils.misc import should_verify_checksum
from polaris.utils.types import (
    AccessType,
    ChecksumStrategy,
    HubOwner,
    IOMode,
    SupportedLicenseType,
    TimeoutTypes,
    ZarrConflictResolution,
)

_HTTPX_SSL_ERROR_CODE = "[SSL: CERTIFICATE_VERIFY_FAILED]"


class PolarisHubClient(OAuth2Client):
    """
    A client for the Polaris Hub API. The Polaris Hub is a central repository of datasets, benchmarks and results.
    Visit it here: [https://polarishub.io/](https://polarishub.io/).

    Bases the [`authlib` client](https://docs.authlib.org/en/latest/client/api.html#authlib.integrations.httpx_client.OAuth2Client),
    which in turns bases the [`httpx` client](https://www.python-httpx.org/advanced/#client-instances).
    See the relevant docs to learn more about how to use these clients outside of the integration with the Polaris Hub.

    Note: Closing the client
        The client should be closed after all requests have been made. For convenience, you can also use the
        client as a context manager to automatically close the client when the context is exited. Note that once
        the client has been closed, it cannot be used anymore.

        ```python
        # Make sure to close the client once finished
        client = PolarisHubClient()
        client.get(...)
        client.close()

        # Or use the client as a context manager
        with PolarisHubClient() as client:
            client.get(...)
        ```

    Note: Interacting with artifacts owned by an organization
        Soon after being added to a new organization on Polaris, there may be a delay spanning some
        minutes where you cannot upload/download artifacts where the aforementioned organization is the owner.
        If this occurs, please re-login via `polaris login --overwrite` and try again.

    Info: Async Client
        `authlib` also supports an [async client](https://docs.authlib.org/en/latest/client/httpx.html#async-oauth-2-0).
        Since we don't expect to make multiple requests to the Hub in parallel
        and due to the added complexity stemming from using the Python asyncio API,
        we are sticking to the sync client - at least for now.
    """

    def __init__(
        self,
        settings: PolarisHubSettings | None = None,
        cache_auth_token: bool = True,
        **kwargs: dict,
    ):
        """
        Args:
            settings: A `PolarisHubSettings` instance.
            cache_auth_token: Whether to cache the auth token to a file.
            **kwargs: Additional keyword arguments passed to the authlib `OAuth2Client` constructor.
        """
        self.settings = PolarisHubSettings() if settings is None else settings

        # We cache the auth token by default, but allow the user to disable this.
        self.token_auth_class = CachedTokenAuth if cache_auth_token else TokenAuth

        super().__init__(
            # OAuth2Client
            token_endpoint=self.settings.hub_token_url,
            token_endpoint_auth_method="none",
            grant_type="urn:ietf:params:oauth:grant-type:token-exchange",
            # httpx.Client
            base_url=self.settings.api_url,
            cert=self.settings.ca_bundle,
            timeout=self.settings.default_timeout,
            # Extra
            **kwargs,
        )

        # We use an external client to get an auth token that can be exchanged for a Polaris Hub token
        self.external_client = ExternalAuthClient(
            settings=self.settings, cache_auth_token=cache_auth_token, **kwargs
        )

    def _prepare_token_endpoint_body(self, body, grant_type, **kwargs):
        """
        Override to support required fields for the token exchange grant type.
        See https://datatracker.ietf.org/doc/html/rfc8693#name-request
        """
        if grant_type == "urn:ietf:params:oauth:grant-type:token-exchange":
            kwargs.update(
                {
                    "subject_token": self.external_client.token["access_token"],
                    "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    "requested_token_type": "urn:ietf:params:oauth:token-type:jwt",
                }
            )
        return super()._prepare_token_endpoint_body(body, grant_type, **kwargs)

    def ensure_active_token(self, token: OAuth2Token) -> bool:
        """
        Override the active check to trigger a refetch of the token if it is not active.
        """
        is_active = super().ensure_active_token(token)
        if is_active:
            return True

        # Check if external token is still valid
        if not self.external_client.ensure_active_token(self.external_client.token):
            return False

        # If so, use it to get a new Hub token
        self.token = self.fetch_token()
        return True

    def _load_from_signed_url(self, url: URLTypes, load_fn: Callable, headers: HeaderTypes | None = None):
        """Utility function to load a file from a signed URL"""
        response = self.get(url, auth=None, headers=headers)  # type: ignore
        response.raise_for_status()
        content = BytesIO(response.content)
        return load_fn(content)

    def _base_request_to_hub(self, url: str, method: str, **kwargs):
        """Utility function since most API methods follow the same pattern"""
        response = self.request(url=url, method=method, **kwargs)

        try:
            response.raise_for_status()

        except HTTPStatusError as error:
            response_status_code = response.status_code

            # With an internal server error, we are not sure the custom error-handling code on the hub is reached.
            if response_status_code == 500:
                raise

            # If JSON is included in the response body, we retrieve it and format it for output. If not, we fallback to
            # retrieving plain text from the body. This is important for handling certain errors thrown from the backend
            # which do not contain JSON in the response.
            try:
                response = response.json()
                response = json.dumps(response, indent=2, sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                response = response.text

            # The below two error cases can happen due to the JWT token containing outdated information.
            # We therefore throw a custom error with a recommended next step.
            if response_status_code == 403:
                # This happens when trying to create an artifact for an owner the user has no access to.
                raise PolarisCreateArtifactError(response=response) from error

            if response_status_code == 404:
                # This happens when an artifact doesn't exist _or_ when the user has no access to that artifact.
                raise PolarisRetrieveArtifactError(response=response) from error

            raise PolarisHubError(response=response) from error
        # Convert the response to json format if the response contains a 'text' body
        try:
            response = response.json()
        except json.JSONDecodeError:
            pass

        return response

    def get_metadata_from_response(self, response: Response, key: str) -> str | None:
        """Get custom metadata saved to the R2 object from the headers."""
        key = f"{self.settings.custom_metadata_prefix}{key}"
        return response.headers.get(key)

    def request(self, method, url, withhold_token=False, auth=httpx.USE_CLIENT_DEFAULT, **kwargs):
        """Wraps the base request method to handle errors"""
        try:
            return super().request(method, url, withhold_token, auth, **kwargs)
        except httpx.ConnectError as error:
            # NOTE (cwognum): In the stack-trace, the more specific SSLCertVerificationError is raised.
            #   We could use the traceback module to cactch this specific error, but that would be overkill here.
            if _HTTPX_SSL_ERROR_CODE in str(error):
                raise ssl.SSLCertVerificationError(
                    "We could not verify the SSL certificate. "
                    f"Please ensure the installed version ({certifi.__version__}) of the `certifi` package is the latest. "
                    "If you require the usage of a custom CA bundle, you can set the POLARIS_CA_BUNDLE "
                    "environment variable to the path of your CA bundle. For debugging, you can temporarily disable "
                    "SSL verification by setting the POLARIS_CA_BUNDLE environment variable to `false`."
                ) from error
            raise error
        except (MissingTokenError, InvalidTokenError, httpx.HTTPStatusError, OAuthError) as error:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code != 401:
                raise
            raise PolarisUnauthorizedError(response=error.response) from error

    def login(self, overwrite: bool = False, auto_open_browser: bool = True):
        """Login to the Polaris Hub using the OAuth2 protocol.

        Warning: Headless authentication
            It is currently not possible to login to the Polaris Hub without a browser.
            See [this Github issue](https://github.com/polaris-hub/polaris/issues/30) for more info.

        Args:
            overwrite: Whether to overwrite the current token if the user is already logged in.
            auto_open_browser: Whether to automatically open the browser to visit the authorization URL.
        """
        if overwrite or self.token is None or not self.ensure_active_token(self.token):
            self.external_client.interactive_login(overwrite=overwrite, auto_open_browser=auto_open_browser)
            self.token = self.fetch_token()

        logger.success("You are successfully logged in to the Polaris Hub.")

    # =========================
    #     API Endpoints
    # =========================

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List all available datasets on the Polaris Hub.

        Args:
            limit: The maximum number of datasets to return.
            offset: The offset from which to start returning datasets.

        Returns:
            A list of dataset names in the format `owner/dataset_name`.
        """
        with ProgressIndicator(
            start_msg="Fetching datasets...",
            success_msg="Fetched datasets.",
            error_msg="Failed to fetch datasets.",
        ):
            response = self._base_request_to_hub(
                url="/dataset", method="GET", params={"limit": limit, "offset": offset}
            )
            dataset_list = [bm["artifactId"] for bm in response["data"]]

            return dataset_list

    def get_dataset(
        self,
        owner: str | HubOwner,
        name: str,
        verify_checksum: ChecksumStrategy = "verify_unless_zarr",
    ) -> Dataset:
        """Load a dataset from the Polaris Hub.

        Args:
            owner: The owner of the dataset. Can be either a user or organization from the Polaris Hub.
            name: The name of the dataset.
            verify_checksum: Whether to use the checksum to verify the integrity of the dataset. If None,
                will infer a practical default based on the dataset's storage location.

        Returns:
            A `Dataset` instance, if it exists.
        """

        with ProgressIndicator(
            start_msg="Fetching dataset...",
            success_msg="Fetched dataset.",
            error_msg="Failed to fetch dataset.",
        ):
            response = self._base_request_to_hub(url=f"/dataset/{owner}/{name}", method="GET")
            storage_response = self.get(response["tableContent"]["url"])

            # This should be a 307 redirect with the signed URL
            if storage_response.status_code != 307:
                try:
                    storage_response.raise_for_status()
                except HTTPStatusError as error:
                    raise PolarisHubError(
                        message="Could not get signed URL from Polaris Hub.", response=storage_response
                    ) from error

            storage_response = storage_response.json()
            url = storage_response["url"]
            headers = storage_response["headers"]

            response["table"] = self._load_from_signed_url(url=url, headers=headers, load_fn=pd.read_parquet)

            dataset = Dataset(**response)

            if should_verify_checksum(verify_checksum, dataset):
                dataset.verify_checksum()
            else:
                dataset.md5sum = response["md5Sum"]

            return dataset

    def open_zarr_file(
        self, owner: str | HubOwner, name: str, path: str, mode: IOMode, as_consolidated: bool = True
    ) -> zarr.hierarchy.Group:
        """Open a Zarr file from a Polaris dataset

        Args:
            owner: Which Hub user or organization owns the artifact.
            name: Name of the dataset.
            path: Path to the Zarr file within the dataset.
            mode: The mode in which the file is opened.
            as_consolidated: Whether to open the store with consolidated metadata for optimized reading.
                This is only applicable in 'r' and 'r+' modes.

        Returns:
            The Zarr object representing the dataset.
        """
        if as_consolidated and mode not in ["r", "r+"]:
            raise ValueError("Consolidated archives can only be used with 'r' or 'r+' mode.")

        polaris_fs = PolarisFileSystem(
            polaris_client=self,
            dataset_owner=owner,
            dataset_name=name,
        )

        try:
            store = zarr.storage.FSStore(path, fs=polaris_fs)
            if mode in ["r", "r+"] and as_consolidated:
                return zarr.open_consolidated(store, mode=mode)
            return zarr.open(store, mode=mode)

        except HTTPStatusError as error:
            # In this case, we can pass the response to provide more information
            raise PolarisHubError(message="Error opening Zarr store", response=error.response) from error
        except Exception as error:
            raise PolarisHubError(message="Error opening Zarr store") from error

    def list_benchmarks(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List all available benchmarks on the Polaris Hub.

        Args:
            limit: The maximum number of benchmarks to return.
            offset: The offset from which to start returning benchmarks.

        Returns:
            A list of benchmark names in the format `owner/benchmark_name`.
        """
        with ProgressIndicator(
            start_msg="Fetching benchmarks...",
            success_msg="Fetched benchmarks.",
            error_msg="Failed to fetch benchmarks.",
        ):
            # TODO (cwognum): What to do with pagination, i.e. limit and offset?
            response = self._base_request_to_hub(
                url="/benchmark", method="GET", params={"limit": limit, "offset": offset}
            )
            benchmarks_list = [f"{HubOwner(**bm['owner'])}/{bm['name']}" for bm in response["data"]]

            return benchmarks_list

    def get_benchmark(
        self,
        owner: str | HubOwner,
        name: str,
        verify_checksum: ChecksumStrategy = "verify_unless_zarr",
    ) -> BenchmarkSpecification:
        """Load a benchmark from the Polaris Hub.

        Args:
            owner: The owner of the benchmark. Can be either a user or organization from the Polaris Hub.
            name: The name of the benchmark.
            verify_checksum: Whether to use the checksum to verify the integrity of the benchmark.

        Returns:
            A `BenchmarkSpecification` instance, if it exists.
        """
        with ProgressIndicator(
            start_msg="Fetching benchmark...",
            success_msg="Fetched benchmark.",
            error_msg="Failed to fetch benchmark.",
        ):
            response = self._base_request_to_hub(url=f"/benchmark/{owner}/{name}", method="GET")

            # TODO (jstlaurent): response["dataset"]["artifactId"] is the owner/name unique identifier,
            #  but we'd need to change the signature of get_dataset to use it
            response["dataset"] = self.get_dataset(
                response["dataset"]["owner"]["slug"],
                response["dataset"]["name"],
                verify_checksum=verify_checksum,
            )

            # TODO (cwognum): As we get more complicated benchmarks, how do we still find the right subclass?
            #  Maybe through structural pattern matching, introduced in Py3.10, or Pydantic's discriminated unions?
            benchmark_cls = (
                SingleTaskBenchmarkSpecification
                if len(response["targetCols"]) == 1
                else MultiTaskBenchmarkSpecification
            )

            benchmark = benchmark_cls(**response)

            if should_verify_checksum(verify_checksum, benchmark.dataset):
                benchmark.verify_checksum()
            else:
                benchmark.md5sum = response["md5Sum"]

            return benchmark

    def upload_results(
        self,
        results: BenchmarkResults,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
    ):
        """Upload the results to the Polaris Hub.

        Info: Owner
            The owner of the results will automatically be inferred by the hub from the user making the request.
            Contrary to other artifact types, an organization cannot own a set of results.
            However, you can specify the `BenchmarkResults.contributors` field to share credit with other hub users.

        Note: Required meta-data
            The Polaris client and hub maintain different requirements as to which meta-data is required.
            The requirements by the hub are stricter, so when uploading to the hub you might
            get some errors on missing meta-data. Make sure to fill-in as much of the meta-data as possible
            before uploading.

        Note: Benchmark name and owner
            Importantly, `results.benchmark_name` and `results.benchmark_owner` must be specified
            and match an existing benchmark on the Polaris Hub. If these results were generated by
            `benchmark.evaluate(...)`, this is done automatically.

        Args:
            results: The results to upload.
            access: Grant public or private access to result
            owner: Which Hub user or organization owns the artifact. Takes precedence over `results.owner`.
        """
        with ProgressIndicator(
            start_msg="Uploading result...",
            success_msg="Uploaded result.",
            error_msg="Failed to upload result.",
        ) as progress_indicator:
            # Get the serialized model data-structure
            results.owner = HubOwner.normalize(owner or results.owner)
            result_json = results.model_dump(by_alias=True, exclude_none=True)

            # Make a request to the hub
            response = self._base_request_to_hub(
                url="/result", method="POST", json={"access": access, **result_json}
            )

            # Inform the user about where to find their newly created artifact.
            result_url = urljoin(
                self.settings.hub_url,
                f"benchmarks/{results.benchmark_owner}/{results.benchmark_name}/{response['id']}",
            )

            progress_indicator.update_success_msg(
                f"Your result has been successfully uploaded to the Hub. View it here: {result_url}"
            )

            return response

    def upload_dataset(
        self,
        dataset: Dataset,
        access: AccessType = "private",
        timeout: TimeoutTypes = (10, 200),
        owner: HubOwner | str | None = None,
        if_exists: ZarrConflictResolution = "replace",
    ):
        """Upload the dataset to the Polaris Hub.

        Info: Owner
            You have to manually specify the owner in the dataset data model. Because the owner could
            be a user or an organization, we cannot automatically infer this from just the logged-in user.

        Note: Required meta-data
            The Polaris client and hub maintain different requirements as to which meta-data is required.
            The requirements by the hub are stricter, so when uploading to the hub you might
            get some errors on missing meta-data. Make sure to fill-in as much of the meta-data as possible
            before uploading.

        Args:
            dataset: The dataset to upload.
            access: Grant public or private access to result
            timeout: Request timeout values. User can modify the value when uploading large dataset as needed.
                This can be a single value with the timeout in seconds for all IO operations, or a more granular
                tuple with (connect_timeout, write_timeout). The type of the the timout parameter comes from `httpx`.
                Since datasets can get large, it might be needed to increase the write timeout for larger datasets.
                See also: https://www.python-httpx.org/advanced/#timeout-configuration
            owner: Which Hub user or organization owns the artifact. Takes precedence over `dataset.owner`.
            if_exists: Action for handling existing files in the Zarr archive. Options are 'raise' to throw
                an error, 'replace' to overwrite, or 'skip' to proceed without altering the existing files.
        """
        with ProgressIndicator(
            start_msg="Uploading dataset...",
            success_msg="Uploaded dataset.",
            error_msg="Failed to upload dataset.",
        ) as progress_indicator:
            # Check if a dataset license was specified prior to upload
            if not dataset.license:
                raise InvalidDatasetError(
                    f"\nPlease specify a supported license for this dataset prior to uploading to the Polaris Hub.\nOnly some licenses are supported - {get_args(SupportedLicenseType)}."
                )

            # Normalize timeout
            if timeout is None:
                timeout = self.settings.default_timeout

            # Get the serialized data-model
            # We exclude the table as it handled separately and we exclude the cache_dir as it is user-specific
            dataset.owner = HubOwner.normalize(owner or dataset.owner)
            dataset_json = dataset.model_dump(
                exclude={"cache_dir", "table"}, exclude_none=True, by_alias=True
            )

            # If the dataset uses Zarr, we will save the Zarr archive to the Hub as well
            if dataset.uses_zarr:
                dataset_json["zarrRootPath"] = f"{PolarisFileSystem.protocol}://data.zarr"

            # Uploading a dataset is a three-step process.
            # 1. Upload the dataset meta data to the hub and prepare the hub to receive the data
            # 2. Upload the parquet file to the hub
            # 3. Upload the associated Zarr archive
            # TODO: Revert step 1 in case step 2 fails - Is this needed? Or should this be taken care of by the hub?

            # Prepare the parquet file
            buffer = BytesIO()
            dataset.table.to_parquet(buffer, engine="auto")
            parquet_size = len(buffer.getbuffer())
            parquet_md5 = md5(buffer.getbuffer()).hexdigest()

            # Step 1: Upload meta-data
            # Instead of directly uploading the data, we announce to the hub that we intend to upload it.
            # We do so separately for the Zarr archive and Parquet file.
            url = f"/dataset/{dataset.owner}/{dataset.name}"
            response = self._base_request_to_hub(
                url=url,
                method="PUT",
                json={
                    "tableContent": {
                        "size": parquet_size,
                        "fileType": "parquet",
                        "md5Sum": parquet_md5,
                    },
                    "zarrContent": [md5sum.model_dump() for md5sum in dataset._zarr_md5sum_manifest],
                    "access": access,
                    **dataset_json,
                },
                timeout=timeout,
            )

            # Step 2: Upload the parquet file
            # create an empty PUT request to get the table content URL from cloudflare
            hub_response = self.request(
                url=response["tableContent"]["url"],
                method="PUT",
                headers={
                    "Content-type": "application/vnd.apache.parquet",
                },
                timeout=timeout,
            )

            if hub_response.status_code == 307:
                # If the hub returns a 307 redirect, we need to follow it to get the signed URL
                hub_response_body = hub_response.json()

                # Upload the data to the cloudflare url
                bucket_response = self.request(
                    url=hub_response_body["url"],
                    method=hub_response_body["method"],
                    headers={
                        "Content-type": "application/vnd.apache.parquet",
                        **hub_response_body["headers"],
                    },
                    content=buffer.getvalue(),
                    auth=None,
                    timeout=timeout,  # required for large size dataset
                )
                bucket_response.raise_for_status()

            else:
                hub_response.raise_for_status()

            # Step 3: Upload any associated Zarr archive
            if dataset.uses_zarr:
                with tmp_attribute_change(self.settings, "default_timeout", timeout):
                    # Copy the Zarr archive to the hub
                    dest = self.open_zarr_file(
                        owner=dataset.owner,
                        name=dataset.name,
                        path=dataset_json["zarrRootPath"],
                        mode="w",
                        as_consolidated=False,
                    )

                    # Locally consolidate Zarr archive metadata. Future updates on handling consolidated
                    # metadata based on Zarr developers' recommendations can be tracked at:
                    # https://github.com/zarr-developers/zarr-python/issues/1731
                    zarr.consolidate_metadata(dataset.zarr_root.store.store)
                    zmetadata_content = dataset.zarr_root.store.store[".zmetadata"]
                    dest.store[".zmetadata"] = zmetadata_content

                    logger.info("Copying Zarr archive to the Hub. This may take a while.")
                    zarr.copy_store(
                        source=dataset.zarr_root.store.store,
                        dest=dest.store,
                        log=logger.debug,
                        if_exists=if_exists,
                    )

            progress_indicator.update_success_msg(
                "Your dataset has been successfully uploaded to the Hub. "
                f"View it here: {urljoin(self.settings.hub_url, f'datasets/{dataset.owner}/{dataset.name}')}"
            )

            return response

    def upload_benchmark(
        self,
        benchmark: BenchmarkSpecification,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
    ):
        """Upload the benchmark to the Polaris Hub.

        Info: Owner
            You have to manually specify the owner in the benchmark data model. Because the owner could
            be a user or an organization, we cannot automatically infer this from the logged-in user.

        Note: Required meta-data
            The Polaris client and hub maintain different requirements as to which meta-data is required.
            The requirements by the hub are stricter, so when uploading to the hub you might
            get some errors on missing meta-data. Make sure to fill-in as much of the meta-data as possible
            before uploading.

        Note: Non-existent datasets
            The client will _not_ upload the associated dataset to the hub if it does not yet exist.
            Make sure to specify an existing dataset or upload the dataset first.

        Args:
            benchmark: The benchmark to upload.
            access: Grant public or private access to result
            owner: Which Hub user or organization owns the artifact. Takes precedence over `benchmark.owner`.
        """
        with ProgressIndicator(
            start_msg="Uploading benchmark...",
            success_msg="Uploaded benchmark.",
            error_msg="Failed to upload benchmark.",
        ) as progress_indicator:
            # Get the serialized data-model
            # We exclude the dataset as we expect it to exist on the hub already.
            benchmark.owner = HubOwner.normalize(owner or benchmark.owner)
            benchmark_json = benchmark.model_dump(exclude={"dataset"}, exclude_none=True, by_alias=True)
            benchmark_json["datasetArtifactId"] = benchmark.dataset.artifact_id
            benchmark_json["access"] = access

            url = f"/benchmark/{benchmark.owner}/{benchmark.name}"
            response = self._base_request_to_hub(url=url, method="PUT", json=benchmark_json)

            progress_indicator.update_success_msg(
                "Your benchmark has been successfully uploaded to the Hub. "
                f"View it here: {urljoin(self.settings.hub_url, f'benchmarks/{benchmark.owner}/{benchmark.name}')}"
            )

            return response
