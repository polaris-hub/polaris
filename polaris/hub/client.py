import json
import logging
from hashlib import md5
from io import BytesIO
from typing import get_args
from urllib.parse import urljoin

import httpx
import pandas as pd
import zarr
from authlib.integrations.base_client.errors import InvalidTokenError, MissingTokenError
from authlib.integrations.httpx_client import OAuth2Client, OAuthError
from authlib.oauth2 import OAuth2Error, TokenAuth
from authlib.oauth2.rfc6749 import OAuth2Token
from httpx import ConnectError, HTTPStatusError, Response
from typing_extensions import Self

from polaris.benchmark import (
    BenchmarkV1Specification,
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.benchmark._benchmark_v2 import BenchmarkV2Specification
from polaris.competition import CompetitionSpecification
from polaris.model import Model
from polaris.dataset import Dataset, DatasetV1, DatasetV2
from polaris.evaluate import BenchmarkResultsV1, BenchmarkResultsV2, CompetitionPredictions
from polaris.hub.external_client import ExternalAuthClient
from polaris.hub.oauth import CachedTokenAuth
from polaris.hub.settings import PolarisHubSettings
from polaris.hub.storage import StorageSession
from polaris.utils.context import track_progress
from polaris.utils.errors import (
    InvalidDatasetError,
    PolarisCreateArtifactError,
    PolarisHubError,
    PolarisRetrieveArtifactError,
    PolarisSSLError,
    PolarisUnauthorizedError,
)
from polaris.utils.types import (
    AccessType,
    ChecksumStrategy,
    HubOwner,
    SupportedLicenseType,
    TimeoutTypes,
    ZarrConflictResolution,
)

logger = logging.getLogger(__name__)

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

    def __enter__(self: Self) -> Self:
        """
        When used as a context manager, automatically check that authentication is valid.
        """
        super().__enter__()
        if not self.ensure_active_token():
            raise PolarisUnauthorizedError()
        return self

    @property
    def has_user_password(self) -> bool:
        return bool(self.settings.username and self.settings.password)

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

    def ensure_active_token(self, token: OAuth2Token | None = None) -> bool:
        """
        Override the active check to trigger a refetch of the token if it is not active.
        """
        # This won't be needed with if we set a lower bound for authlib: >=1.3.2
        # See https://github.com/lepture/authlib/pull/625
        # As of now, this latest version is not available on Conda though.
        token = token or self.token
        is_active = super().ensure_active_token(token) if token else False
        if is_active:
            return True

        # Check if external token is still valid, or we're using password auth
        if not (self.has_user_password or self.external_client.ensure_active_token()):
            return False

        # If so, use it to get a new Hub token
        self.token = self.fetch_token()
        return True

    def fetch_token(self, **kwargs):
        """
        Handles the optional support for password grant type, and provide better error messages.
        """
        try:
            return super().fetch_token(
                username=self.settings.username,
                password=self.settings.password,
                grant_type="password"
                if self.has_user_password
                else "urn:ietf:params:oauth:grant-type:token-exchange",
                **kwargs,
            )
        except (OAuthError, OAuth2Error) as error:
            raise PolarisHubError(
                message=f"Could not obtain a token to access the Hub. Error was: {error.error} - {error.description}"
            ) from error

    def _base_request_to_hub(self, url: str, method: str, **kwargs) -> Response:
        """Utility function since most API methods follow the same pattern"""
        try:
            response = self.request(url=url, method=method, **kwargs)
            response.raise_for_status()
            return response
        except HTTPStatusError as error:
            # If JSON is included in the response body, we retrieve it and format it for output. If not, we fall back to
            # retrieving plain text from the body. 500 errors will not have a JSON response.
            try:
                response_text = error.response.json()
                response_text = json.dumps(response_text, indent=2, sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                response_text = error.response.text

            match error.response.status_code:
                case 401:
                    # Providing the user a more helpful error message suggesting a re-login when their access credentials expire.
                    raise PolarisUnauthorizedError(response_text=response_text) from error
                case 403:
                    # The error can happen due to the JWT token containing outdated information.
                    raise PolarisCreateArtifactError(response_text=response_text) from error
                case 404:
                    # This happens when an artifact doesn't exist _or_ when the user has no access to that artifact.
                    raise PolarisRetrieveArtifactError(response_text=response_text) from error
                case _:
                    raise PolarisHubError(response_text=response_text) from error

    def get_metadata_from_response(self, response: Response, key: str) -> str | None:
        """Get custom metadata saved to the R2 object from the headers."""
        key = f"{self.settings.custom_metadata_prefix}{key}"
        return response.headers.get(key)

    def request(self, method, url, withhold_token=False, auth=httpx.USE_CLIENT_DEFAULT, **kwargs):
        """Wraps the base request method to handle errors"""
        try:
            return super().request(method, url, withhold_token, auth, **kwargs)
        except ConnectError as error:
            # NOTE (cwognum): In the stack trace, the more specific SSLCertVerificationError is raised.
            #   We could use the traceback module to catch this specific error, but that would be overkill here.
            error_string = str(error)
            ExceptionClass = PolarisSSLError if _HTTPX_SSL_ERROR_CODE in error_string else PolarisHubError
            raise ExceptionClass(error_string) from error
        except (MissingTokenError, InvalidTokenError, OAuthError) as error:
            raise PolarisUnauthorizedError() from error

    def login(self, overwrite: bool = False, auto_open_browser: bool = True):
        """Login to the Polaris Hub using the OAuth2 protocol.

        Warning: Headless authentication
            It is currently not possible to login to the Polaris Hub without a browser.
            See [this Github issue](https://github.com/polaris-hub/polaris/issues/30) for more info.

        Args:
            overwrite: Whether to overwrite the current token if the user is already logged in.
            auto_open_browser: Whether to automatically open the browser to visit the authorization URL.
        """
        if overwrite or self.token is None or not self.ensure_active_token():
            self.external_client.interactive_login(overwrite=overwrite, auto_open_browser=auto_open_browser)
            self.token = self.fetch_token()

        logger.info("You are successfully logged in to the Polaris Hub.")

    # =========================
    #     API Endpoints
    # =========================

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List all available datasets (v1 and v2) on the Polaris Hub.
        We prioritize v2 datasets over v1 datasets.

        Args:
            limit: The maximum number of datasets to return.
            offset: The offset from which to start returning datasets.

        Returns:
            A list of dataset names in the format `owner/dataset_slug`.
        """
        with track_progress(description="Fetching datasets", total=1):
            # Step 1: Fetch enough v2 datasets to cover the offset and limit
            v2_json_response = self._base_request_to_hub(
                url="/v2/dataset", method="GET", params={"limit": limit, "offset": offset}
            ).json()
            v2_data = v2_json_response["data"]
            v2_datasets = [dataset["artifactId"] for dataset in v2_data]

            # If v2 datasets satisfy the limit, return them
            if len(v2_datasets) == limit:
                return v2_datasets

            # Step 2: Calculate the remaining limit and fetch v1 datasets
            remaining_limit = max(0, limit - len(v2_datasets))

            v1_json_response = self._base_request_to_hub(
                url="/v1/dataset",
                method="GET",
                params={
                    "limit": remaining_limit,
                    "offset": max(0, offset - v2_json_response["metadata"]["total"]),
                },
            ).json()
            v1_data = v1_json_response["data"]
            v1_datasets = [dataset["artifactId"] for dataset in v1_data]

            # Combine the v2 and v1 datasets
            combined_datasets = v2_datasets + v1_datasets

            # Ensure the final combined list respects the limit
            return combined_datasets

    def get_dataset(
        self,
        owner: str | HubOwner,
        slug: str,
        verify_checksum: ChecksumStrategy = "verify_unless_zarr",
    ) -> DatasetV1 | DatasetV2:
        """Load a standard dataset from the Polaris Hub.

        Args:
            owner: The owner of the dataset. Can be either a user or organization from the Polaris Hub.
            slug: The slug of the dataset.
            verify_checksum: Whether to use the checksum to verify the integrity of the dataset. If None,
                will infer a practical default based on the dataset's storage location.

        Returns:
            A `Dataset` instance, if it exists.
        """
        with track_progress(description="Fetching dataset", total=1):
            try:
                return self._get_v1_dataset(owner, slug, verify_checksum)
            except PolarisRetrieveArtifactError:
                # If the v1 dataset is not found, try to load a v2 dataset
                return self._get_v2_dataset(owner, slug)

    def _get_v1_dataset(
        self,
        owner: str | HubOwner,
        slug: str,
        verify_checksum: ChecksumStrategy = "verify_unless_zarr",
    ) -> DatasetV1:
        """Loads a V1 dataset from Polaris Hub

        Args:
            owner: The owner of the dataset. Can be either a user or organization from the Polaris Hub.
            slug: The slug of the dataset.
            verify_checksum: Whether to use the checksum to verify the integrity of the dataset.

        Returns:
            A `Dataset` instance, if it exists.
        """
        url = f"/v1/dataset/{owner}/{slug}"
        response = self._base_request_to_hub(url=url, method="GET")
        response_data = response.json()

        # Disregard the Zarr root in the response. We'll get it from the storage token instead.
        response_data.pop("zarrRootPath", None)

        # Load the dataset table and optional Zarr archive
        with StorageSession(self, "read", Dataset.urn_for(owner, slug)) as storage:
            table = pd.read_parquet(BytesIO(storage.get_file("root")))
            zarr_root_path = storage.paths.extension

            if zarr_root_path is not None:
                # For V1 datasets, the Zarr Root is optional.
                # It should be None if the dataset does not use pointer columns
                zarr_root_path = str(zarr_root_path)

        dataset = DatasetV1(table=table, zarr_root_path=zarr_root_path, **response_data)
        md5sum = response_data["md5Sum"]

        if dataset.should_verify_checksum(verify_checksum):
            dataset.verify_checksum(md5sum)
        else:
            dataset.md5sum = md5sum

        return dataset

    def _get_v2_dataset(self, owner: str | HubOwner, slug: str) -> DatasetV2:
        """"""
        url = f"/v2/dataset/{owner}/{slug}"
        response = self._base_request_to_hub(url=url, method="GET")
        response_data = response.json()

        # Disregard the Zarr root in the response. We'll get it from the storage token instead.
        response_data.pop("zarrRootPath", None)

        # Load the Zarr archive
        with StorageSession(self, "read", DatasetV2.urn_for(owner, slug)) as storage:
            zarr_root_path = str(storage.paths.root)

        dataset = DatasetV2(zarr_root_path=zarr_root_path, **response_data)
        return dataset

    def list_benchmarks(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List all available benchmarks (v1 and v2) on the Polaris Hub.
        We prioritize v2 benchmarks over v1 benchmarks.

        Args:
            limit: The maximum number of benchmarks to return.
            offset: The offset from which to start returning benchmarks.

        Returns:
            A list of benchmark names in the format `owner/benchmark_slug`.
        """
        with track_progress(description="Fetching benchmarks", total=1):
            # Step 1: Fetch enough v2 benchmarks to cover the offset and limit
            v2_json_response = self._base_request_to_hub(
                url="/v2/benchmark", method="GET", params={"limit": limit, "offset": offset}
            ).json()
            v2_data = v2_json_response["data"]
            v2_benchmarks = [benchmark["artifactId"] for benchmark in v2_data]

            # If v2 benchmarks satisfy the limit, return them
            if len(v2_benchmarks) == limit:
                return v2_benchmarks

            # Step 2: Calculate the remaining limit and fetch v1 benchmarks
            remaining_limit = max(0, limit - len(v2_benchmarks))

            v1_json_response = self._base_request_to_hub(
                url="/v1/benchmark",
                method="GET",
                params={
                    "limit": remaining_limit,
                    "offset": max(0, offset - v2_json_response["metadata"]["total"]),
                },
            ).json()
            v1_data = v1_json_response["data"]
            v1_benchmarks = [benchmark["artifactId"] for benchmark in v1_data]

            # Combine the v2 and v1 benchmarks
            combined_benchmarks = v2_benchmarks + v1_benchmarks

            # Ensure the final combined list respects the limit
            return combined_benchmarks

    def get_benchmark(
        self,
        owner: str | HubOwner,
        slug: str,
        verify_checksum: ChecksumStrategy = "verify_unless_zarr",
    ) -> BenchmarkV1Specification | BenchmarkV2Specification:
        """Load a benchmark from the Polaris Hub.

        Args:
            owner: The owner of the benchmark. Can be either a user or organization from the Polaris Hub.
            slug: The slug of the benchmark.
            verify_checksum: Whether to use the checksum to verify the integrity of the benchmark.

        Returns:
            A `BenchmarkSpecification` instance, if it exists.
        """
        with track_progress(description="Fetching benchmark", total=1):
            try:
                return self._get_v1_benchmark(owner, slug, verify_checksum)
            except PolarisRetrieveArtifactError:
                # If the v1 benchmark is not found, try to load a v2 benchmark
                return self._get_v2_benchmark(owner, slug)

    def _get_v1_benchmark(
        self,
        owner: str | HubOwner,
        slug: str,
        verify_checksum: ChecksumStrategy = "verify_unless_zarr",
    ) -> BenchmarkV1Specification:
        response = self._base_request_to_hub(url=f"/v1/benchmark/{owner}/{slug}", method="GET")
        response_data = response.json()

        # TODO (jstlaurent): response["dataset"]["artifactId"] is the owner/name unique identifier,
        #  but we'd need to change the signature of get_dataset to use it
        response_data["dataset"] = self.get_dataset(
            response_data["dataset"]["owner"]["slug"],
            response_data["dataset"]["name"],
            verify_checksum=verify_checksum,
        )

        # TODO (cwognum): As we get more complicated benchmarks, how do we still find the right subclass?
        #  Maybe through structural pattern matching, introduced in Py3.10, or Pydantic's discriminated unions?
        benchmark_cls = (
            SingleTaskBenchmarkSpecification
            if len(response_data["targetCols"]) == 1
            else MultiTaskBenchmarkSpecification
        )

        benchmark = benchmark_cls(**response_data)

        if benchmark.dataset.should_verify_checksum(verify_checksum):
            benchmark.verify_checksum()
        else:
            benchmark.md5sum = response_data["md5Sum"]

        return benchmark

    def _get_v2_benchmark(self, owner: str | HubOwner, slug: str) -> BenchmarkV2Specification:
        response = self._base_request_to_hub(url=f"/v2/benchmark/{owner}/{slug}", method="GET")
        response_data = response.json()

        response_data["dataset"] = self.get_dataset(*response_data["dataset"]["artifactId"].split("/"))

        # Load the split index sets
        with StorageSession(self, "read", BenchmarkV2Specification.urn_for(owner, slug)) as storage:
            split = {label: storage.get_file(label) for label in response_data.get("split", {}).keys()}

        return BenchmarkV2Specification(**{**response_data, "split": split})

    def upload_results(
        self,
        results: BenchmarkResultsV1 | BenchmarkResultsV2,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
    ):
        """Upload the results to the Polaris Hub.

        Info: Owner
            The owner of the results will automatically be inferred by the Hub from the user making the request.
            Contrary to other artifact types, an organization cannot own a set of results.
            However, you can specify the `BenchmarkResults.contributors` field to share credit with other hub users.

        Note: Required metadata
            The Polaris client and Hub maintain different requirements as to which metadata is required.
            The requirements by the Hub are stricter, so when uploading to the Hub you might
            get some errors on missing metadata. Make sure to fill-in as much of the metadata as possible
            before uploading.

        Args:
            results: The results to upload.
            access: Grant public or private access to result
            owner: Which Hub user or organization owns the artifact. Takes precedence over `results.owner`.
        """
        with track_progress(description="Uploading results", total=1) as (progress, task):
            # Get the serialized model data-structure
            results.owner = HubOwner.normalize(owner or results.owner)
            result_json = results.model_dump(by_alias=True, exclude_none=True)

            # Make a request to the Hub
            response = self._base_request_to_hub(
                url="/v2/result", method="POST", json={"access": access, **result_json}
            )

            # Inform the user about where to find their newly created artifact.
            result_url = urljoin(self.settings.hub_url, response.headers.get("Content-Location"))

            progress.log(
                f"[green]Your result has been successfully uploaded to the Hub. View it here: {result_url}"
            )

    def upload_dataset(
        self,
        dataset: DatasetV1 | DatasetV2,
        access: AccessType = "private",
        timeout: TimeoutTypes = (10, 200),
        owner: HubOwner | str | None = None,
        if_exists: ZarrConflictResolution = "replace",
        parent_artifact_id: str | None = None,
    ):
        """Upload a dataset to the Polaris Hub.

        Info: Owner
            You have to manually specify the owner in the dataset data model. Because the owner could
            be a user or an organization, we cannot automatically infer this from just the logged-in user.

        Note: Required metadata
            The Polaris client and Hub maintain different requirements as to which metadata is required.
            The requirements by the Hub are stricter, so when uploading to the Hub you might
            get some errors on missing metadata. Make sure to fill-in as much of the metadata as possible
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
            parent_artifact_id: The `owner/slug` of the parent dataset, if uploading a new version of a dataset.
        """
        # Normalize timeout
        if timeout is None:
            timeout = self.settings.default_timeout

        # Check if a dataset license was specified prior to upload
        if not dataset.license:
            raise InvalidDatasetError(
                f"\nPlease specify a supported license for this dataset prior to uploading to the Polaris Hub.\nOnly some licenses are supported - {get_args(SupportedLicenseType)}."
            )

        if isinstance(dataset, DatasetV1):
            self._upload_v1_dataset(dataset, timeout, access, owner, if_exists, parent_artifact_id)
        elif isinstance(dataset, DatasetV2):
            self._upload_v2_dataset(dataset, timeout, access, owner, if_exists, parent_artifact_id)

    def _upload_v1_dataset(
        self,
        dataset: DatasetV1,
        timeout: TimeoutTypes,
        access: AccessType,
        owner: HubOwner | str | None,
        if_exists: ZarrConflictResolution,
        parent_artifact_id: str | None,
    ):
        """
        Upload a V1 dataset to the Polaris Hub.
        """

        with track_progress(description="Uploading dataset", total=1) as (progress, task):
            # Get the serialized data-model
            # We exclude the table as it handled separately
            dataset.owner = HubOwner.normalize(owner or dataset.owner)
            dataset_json = dataset.model_dump(exclude={"table"}, exclude_none=True, by_alias=True)

            # If the dataset uses Zarr, we will save the Zarr archive to the Hub as well
            if dataset.uses_zarr:
                dataset_json["zarrRootPath"] = f"{StorageSession.polaris_protocol}://data.zarr"

            # Uploading a dataset is a three-step process.
            # 1. Upload the dataset metadata to the Hub and prepare the Hub to receive the data
            # 2. Upload the parquet file to Hub storage
            # 3. Upload the associated Zarr archive to Hub storage

            # Prepare the parquet file
            in_memory_parquet = BytesIO()
            dataset.table.to_parquet(in_memory_parquet)
            parquet_size = len(in_memory_parquet.getbuffer())
            parquet_md5 = md5(in_memory_parquet.getbuffer()).hexdigest()

            # Step 1: Upload metadata
            # Instead of directly uploading the data, we announce to the Hub that we intend to upload it.
            # We do so separately for the Zarr archive and Parquet file.
            url = f"/v1/dataset/{dataset.artifact_id}"
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
                    "parentArtifactId": parent_artifact_id,
                    **dataset_json,
                },
                timeout=timeout,
            )

            inserted_dataset = response.json()

            # We modify the slug in the server
            # Update dataset.slug here so dataset.urn is constructed correctly
            dataset.slug = inserted_dataset["slug"]

            with StorageSession(self, "write", dataset.urn) as storage:
                with track_progress(description="Copying Parquet file", total=1) as (progress, task):
                    # Step 2: Upload the parquet file
                    progress.log("[yellow]This may take a while.")
                    storage.set_file("root", in_memory_parquet.getvalue())

                    # Step 3: Upload any associated Zarr archive
                if dataset.uses_zarr:
                    with track_progress(description="Copying Zarr archive", total=1):
                        destination = storage.store("extension")

                        # Locally consolidate Zarr archive metadata. Future updates on handling consolidated
                        # metadata based on Zarr developers' recommendations can be tracked at:
                        # https://github.com/zarr-developers/zarr-python/issues/1731
                        zarr.consolidate_metadata(dataset.zarr_root.store.store)
                        zmetadata_content = dataset.zarr_root.store.store[".zmetadata"]
                        destination[".zmetadata"] = zmetadata_content

                        # Copy the Zarr archive to the Hub
                        destination.copy_from_source(
                            dataset.zarr_root.store.store, if_exists=if_exists, log=logger.info
                        )

            dataset_url = urljoin(self.settings.hub_url, response.headers.get("Content-Location"))
            progress.log(
                f"[green]Your dataset has been successfully uploaded to the Hub.\nView it here: {dataset_url}"
            )

    def _upload_v2_dataset(
        self,
        dataset: DatasetV2,
        timeout: TimeoutTypes,
        access: AccessType,
        owner: HubOwner | str | None,
        if_exists: ZarrConflictResolution,
        parent_artifact_id: str | None,
    ):
        """
        Upload a V2 dataset to the Polaris Hub.
        """

        with track_progress(description="Uploading dataset", total=1) as (progress, task):
            # Get the serialized data-model
            dataset.owner = HubOwner.normalize(owner or dataset.owner)
            dataset_json = dataset.model_dump(exclude_none=True, by_alias=True)

            # Step 1: Upload dataset metadata
            url = f"/v2/dataset/{dataset.artifact_id}"
            response = self._base_request_to_hub(
                url=url,
                method="PUT",
                json={
                    "zarrManifestFileContent": {
                        "md5Sum": dataset.zarr_manifest_md5sum,
                    },
                    "access": access,
                    "parentArtifactId": parent_artifact_id,
                    **dataset_json,
                },
                timeout=timeout,
            )

            inserted_dataset = response.json()

            # We modify the slug in the server
            # Update dataset.slug here so dataset.urn is constructed correctly
            dataset.slug = inserted_dataset["slug"]

            with StorageSession(self, "write", dataset.urn) as storage:
                # Step 2: Upload the manifest file
                with track_progress(description="Copying manifest file", total=1):
                    with open(dataset.zarr_manifest_path, "rb") as manifest_file:
                        storage.set_file("manifest", manifest_file.read())

                # Step 3: Upload the Zarr archive
                with track_progress(description="Copying Zarr archive", total=1) as (
                    progress_zarr,
                    task_zarr,
                ):
                    progress_zarr.log("[yellow]This may take a while.")

                    destination = storage.store("root")

                    # Locally consolidate Zarr archive metadata. Future updates on handling consolidated
                    # metadata based on Zarr developers' recommendations can be tracked at:
                    # https://github.com/zarr-developers/zarr-python/issues/1731
                    zarr.consolidate_metadata(dataset.zarr_root.store.store)
                    zmetadata_content = dataset.zarr_root.store.store[".zmetadata"]
                    destination[".zmetadata"] = zmetadata_content

                    # Copy the Zarr archive to the Hub
                    destination.copy_from_source(
                        dataset.zarr_root.store.store, if_exists=if_exists, log=logger.info
                    )

            dataset_url = urljoin(self.settings.hub_url, response.headers.get("Content-Location"))
            progress.log(
                f"[green]Your V2 dataset has been successfully uploaded to the Hub.\nView it here: {dataset_url}"
            )

    def upload_benchmark(
        self,
        benchmark: BenchmarkV1Specification | BenchmarkV2Specification,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
        parent_artifact_id: str | None = None,
    ):
        """Upload a benchmark to the Polaris Hub.

        Info: Owner
            You have to manually specify the owner in the benchmark data model. Because the owner could
            be a user or an organization, we cannot automatically infer this from the logged-in user.

        Note: Required metadata
            The Polaris client and Hub maintain different requirements as to which metadata is required.
            The requirements by the Hub are stricter, so when uploading to the Hub you might
            get some errors on missing metadata. Make sure to fill-in as much of the metadata as possible
            before uploading.

        Note: Non-existent datasets
            The client will _not_ upload the associated dataset to the Hub if it does not yet exist.
            Make sure to specify an existing dataset or upload the dataset first.

        Args:
            benchmark: The benchmark to upload.
            access: Grant public or private access to result
            owner: Which Hub user or organization owns the artifact. Takes precedence over `benchmark.owner`.
            parent_artifact_id: The `owner/slug` of the parent benchmark, if uploading a new version of a benchmark.
        """
        match benchmark:
            case BenchmarkV1Specification():
                self._upload_v1_benchmark(benchmark, access, owner, parent_artifact_id)
            case BenchmarkV2Specification():
                self._upload_v2_benchmark(benchmark, access, owner, parent_artifact_id)

    def _upload_v1_benchmark(
        self,
        benchmark: BenchmarkV1Specification,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
        parent_artifact_id: str | None = None,
    ):
        """
        Upload a V1 benchmark to the Polaris Hub.
        """
        with track_progress(description="Uploading benchmark", total=1) as (progress, task):
            # Get the serialized data-model
            # We exclude the dataset as we expect it to exist on the Hub already.
            benchmark.owner = HubOwner.normalize(owner or benchmark.owner)
            benchmark_json = benchmark.model_dump(exclude={"dataset"}, exclude_none=True, by_alias=True)
            benchmark_json["datasetArtifactId"] = benchmark.dataset.artifact_id
            benchmark_json["access"] = access

            url = f"/v1/benchmark/{benchmark.artifact_id}"
            response = self._base_request_to_hub(
                url=url, method="PUT", json={"parentArtifactId": parent_artifact_id, **benchmark_json}
            )

            benchmark_url = urljoin(self.settings.hub_url, response.headers.get("Content-Location"))
            progress.log(
                f"[green]Your benchmark has been successfully uploaded to the Hub.\nView it here: {benchmark_url}"
            )

    def _upload_v2_benchmark(
        self,
        benchmark: BenchmarkV2Specification,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
        parent_artifact_id: str | None = None,
    ):
        """
        Upload a V2 benchmark to the Polaris Hub.
        """
        with track_progress(description="Uploading benchmark", total=1) as (progress, task):
            # Get the serialized data-model
            # We exclude the dataset as we expect it to exist on the Hub already.
            benchmark.owner = HubOwner.normalize(owner or benchmark.owner)
            benchmark_json = benchmark.model_dump(exclude_none=True, by_alias=True)

            # Uploading a V2 benchmark is a multistep process.
            # 1. Upload the benchmark metadata to the Hub and prepare the Hub to receive the data
            # 2. Upload each index set bitmap to the Hub storage

            # Step 1: Upload metadata
            url = f"/v2/benchmark/{benchmark.artifact_id}"
            response = self._base_request_to_hub(
                url=url,
                method="PUT",
                json={
                    "access": access,
                    "datasetArtifactId": benchmark.dataset.artifact_id,
                    "parentArtifactId": parent_artifact_id,
                    **benchmark_json,
                },
            )

            inserted_benchmark = response.json()

            # We modify the slug in the server
            # Update benchmark.slug here so benchmark.urn is constructed correctly
            benchmark.slug = inserted_benchmark["slug"]

            with StorageSession(self, "write", benchmark.urn) as storage:
                logger.info("Copying the benchmark split to the Hub. This may take a while.")

                # 2. Upload each index set bitmap
                with track_progress(
                    description="Copying index sets", total=benchmark.split.n_test_sets + 1
                ) as (progress_index_sets, task_index_sets):
                    for label, index_set in benchmark.split:
                        logger.info(f"Copying index set {label} to the Hub.")
                        storage.set_file(label, index_set.serialize())
                        progress_index_sets.update(task_index_sets, advance=1, refresh=True)

            benchmark_url = urljoin(self.settings.hub_url, response.headers.get("Content-Location"))
            progress.log(
                f"[green]Your benchmark has been successfully uploaded to the Hub.\nView it here: {benchmark_url}"
            )

    def get_competition(self, artifact_id: str) -> CompetitionSpecification:
        """Load a competition from the Polaris Hub.

        Args:
            artifact_id: The artifact identifier for the competition

        Returns:
            A `CompetitionSpecification` instance, if it exists.
        """
        url = f"/v1/competition/{artifact_id}"
        response = self._base_request_to_hub(url=url, method="GET")
        response_data = response.json()

        with StorageSession(
            self, "read", CompetitionSpecification.urn_for(*artifact_id.split("/"))
        ) as storage:
            zarr_root_path = str(storage.paths.root)

        return CompetitionSpecification(zarr_root_path=zarr_root_path, **response_data)

    def submit_competition_predictions(
        self,
        competition: CompetitionSpecification,
        competition_predictions: CompetitionPredictions,
    ):
        """Submit predictions for a competition to the Polaris Hub. The Hub will evaluate them against
        the secure test set and store the result.

        Args:
            competition: The competition to evaluate the predictions for.
            competition_predictions: The predictions and associated metadata to be submitted to the Hub.
        """
        with track_progress(description="Submitting competition predictions", total=1):
            # Prepare prediction payload for submission
            prediction_json = competition_predictions.model_dump(by_alias=True, exclude_none=True)
            prediction_payload = {
                "competitionArtifactId": f"{competition.artifact_id}",
                **prediction_json,
            }

            # Submit payload to Hub
            response = self._base_request_to_hub(
                url="/v1/competition-prediction",
                method="POST",
                json=prediction_payload,
            )
            return response

    def list_models(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List all available models on the Polaris Hub.

        Args:
            limit: The maximum number of models to return.
            offset: The offset from which to start returning models.

        Returns:
            A list of models names in the format `owner/model_slug`.
        """
        with track_progress(description="Fetching models", total=1):
            json_response = self._base_request_to_hub(
                url="/v2/model", method="GET", params={"limit": limit, "offset": offset}
            ).json()
            models = [model["artifactId"] for model in json_response["data"]]

            return models

    def get_model(self, artifact_id: str) -> Model:
        url = f"/v2/model/{artifact_id}"
        response = self._base_request_to_hub(url=url, method="GET")
        response_data = response.json()

        return Model(**response_data)

    def upload_model(
        self,
        model: Model,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
    ):
        """Upload a model to the Polaris Hub.

        Info: Owner
            You have to manually specify the owner in the model data model. Because the owner could
            be a user or an organization, we cannot automatically infer this from just the logged-in user.

        Note: Required metadata
            The Polaris client and Hub maintain different requirements as to which metadata is required.
            The requirements by the Hub are stricter, so when uploading to the Hub you might
            get some errors on missing metadata. Make sure to fill-in as much of the metadata as possible
            before uploading.

        Args:
            model: The model to upload.
            access: Grant public or private access to result
            owner: Which Hub user or organization owns the artifact. Takes precedence over `model.owner`.
        """
        with track_progress(description="Uploading model", total=1) as (progress, task):
            # Get the serialized model data-structure
            model.owner = HubOwner.normalize(owner or model.owner)
            model_json = model.model_dump(by_alias=True, exclude_none=True)

            # Make a request to the Hub
            url = f"/v2/model/{model.artifact_id}"
            response = self._base_request_to_hub(url=url, method="PUT", json={"access": access, **model_json})

            # Inform the user about where to find their newly created artifact.
            model_url = urljoin(self.settings.hub_url, response.headers.get("Content-Location"))

            progress.log(
                f"[green]Your model has been successfully uploaded to the Hub. View it here: {model_url}"
            )
