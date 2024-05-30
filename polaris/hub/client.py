import json
import ssl
import webbrowser
from hashlib import md5
from io import BytesIO
from typing import Callable, Optional, Union, get_args
from urllib.parse import urljoin

import certifi
import fsspec
import httpx
import pandas as pd
import zarr
from authlib.common.security import generate_token
from authlib.integrations.base_client.errors import InvalidTokenError, MissingTokenError
from authlib.integrations.httpx_client import OAuth2Client, OAuthError
from authlib.oauth2.client import OAuth2Client as _OAuth2Client
from datamol.utils import fs
from httpx import HTTPStatusError
from httpx._types import HeaderTypes, URLTypes
from loguru import logger

from polaris.benchmark import (
    BenchmarkSpecification,
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset import Dataset
from polaris.evaluate import BenchmarkResults
from polaris.hub.polarisfs import PolarisFileSystem
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.context import tmp_attribute_change
from polaris.utils.errors import InvalidDatasetError, PolarisHubError, PolarisUnauthorizedError
from polaris.utils.types import (
    AccessType,
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

    Info: Async Client
        `authlib` also supports an [async client](https://docs.authlib.org/en/latest/client/httpx.html#async-oauth-2-0).
        Since we don't expect to make multiple requests to the Hub in parallel
        and due to the added complexity stemming from using the Python asyncio API,
        we are sticking to the sync client - at least for now.
    """

    def __init__(
        self,
        settings: Optional[PolarisHubSettings] = None,
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

        self.settings = PolarisHubSettings() if settings is None else settings

        # We cache the auth token by default, but allow the user to disable this.
        self.cache_auth_token = cache_auth_token
        token = kwargs.get("token")
        if fs.exists(self._auth_token_cache_path) and token is None:  # type: ignore
            with fsspec.open(self._auth_token_cache_path, "r") as fd:
                token = json.load(fd)  # type: ignore

        verify = self.settings.ca_bundle
        if verify is None:
            verify = True

        self.code_verifier = generate_token(48)

        super().__init__(
            # OAuth2Client
            client_id=self.settings.client_id,
            redirect_uri=self.settings.callback_url,
            scope=self.settings.scopes,
            token=token,
            token_endpoint=self.settings.token_fetch_url,
            code_challenge_method="S256",
            # httpx.Client
            base_url=self.settings.api_url,
            verify=verify,
            timeout=self.settings.default_timeout,
            # Extra
            **kwargs,
        )

    def _load_from_signed_url(self, url: URLTypes, load_fn: Callable, headers: Optional[HeaderTypes] = None):
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
            # With an internal server error, we are not sure the custom error-handling code on the hub is reached.
            if response.status_code == 500:
                raise

            # If not an internal server error, the hub should always return a JSON response
            # with additional information about the error.
            response = response.json()
            response = json.dumps(response, indent=2, sort_keys=True)
            raise PolarisHubError(
                f"The request to the Polaris Hub failed. See the error message below for more details:\n{response}"
            ) from error
        # Convert the response to json format if the response contains a 'text' body
        try:
            response = response.json()
        except json.JSONDecodeError:
            pass

        return response

    def _normalize_owner(
        self,
        artifact_owner: Optional[Union[str, HubOwner]] = None,
        parameter_owner: Optional[Union[str, HubOwner]] = None,
    ) -> HubOwner:
        """
        Normalize the owner of an artifact to a `HubOwner` instance.
        The parameter owner takes precedence over the artifact owner.
        """
        if parameter_owner is not None:
            artifact_owner = parameter_owner

        if artifact_owner is None:
            raise ValueError(
                "Either specify the `owner` attribute for the artifact or pass the `owner` parameter."
            )

        return artifact_owner if isinstance(artifact_owner, HubOwner) else HubOwner(slug=artifact_owner)

    # =========================
    #     Overrides
    # =========================

    @_OAuth2Client.token.setter
    def token(self, value):
        """Override the token setter to additionally save the token to a cache"""
        super(OAuth2Client, type(self)).token.fset(self, value)  # type: ignore

        # We cache afterwards, because the token setter adds fields we need to save (i.e. expires_at).
        if self.cache_auth_token:
            with fsspec.open(self._auth_token_cache_path, "w") as fd:
                json.dump(value, fd)  # type: ignore

    @property
    def _auth_token_cache_path(self) -> str:
        """If self.cache_auth_token is True, this is the location of the cached auth token."""
        path = None
        if self.cache_auth_token:
            fs.mkdir(DEFAULT_CACHE_DIR, exist_ok=True)
            path = fs.join(DEFAULT_CACHE_DIR, "polaris_auth_token.json")
        return path

    def create_authorization_url(self, **kwargs) -> tuple[str, Optional[str]]:
        """Light wrapper to automatically pass in the right URL."""
        return super().create_authorization_url(
            url=self.settings.authorize_url, code_verifier=self.code_verifier, **kwargs
        )

    def fetch_token(self, **kwargs):
        """Light wrapper to automatically pass in the right URL"""
        return super().fetch_token(
            url=self.settings.token_fetch_url, code_verifier=self.code_verifier, **kwargs
        )

    def request(self, method, url, withhold_token=False, auth=httpx.USE_CLIENT_DEFAULT, **kwargs):
        """Wraps the base request method to handle errors"""
        try:
            response = super().request(method, url, withhold_token, auth, **kwargs)
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
            raise PolarisUnauthorizedError from error
        return response

    # =========================
    #     API Endpoints
    # =========================

    @property
    def user_info(self) -> dict:
        """Get information about the currently logged in user through the OAuth2 User Info flow."""

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

    @property
    def user_as_owner(self) -> HubOwner:
        """Easily get the currently logged-in user a `HubOwner` instance."""
        return HubOwner(user_id=self.user_info["user_id"], slug=self.user_info["username"])

    def login(self, overwrite: bool = False, auto_open_browser: bool = True):
        """Login to the Polaris Hub using the OAuth2 protocol.

        Warning: Headless authentication
            It is currently not possible to login to the Polaris Hub without a browser.
            See [this Github issue](https://github.com/polaris-hub/polaris/issues/30) for more info.

        Args:
            overwrite: Whether to overwrite the current token if the user is already logged in.
            auto_open_browser: Whether to automatically open the browser to visit the authorization URL.
        """

        # Check if the user is already logged in
        if self.token is not None and not overwrite:
            try:
                info = self.user_info
                logger.info(
                    f"You are already logged in to the Polaris Hub as {info['username']} ({info['email']}). "
                    "Set `overwrite=True` to force re-authentication."
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

        logger.success(
            f"Successfully authenticated to the Polaris Hub "
            f"as `{self.user_info['username']}` ({self.user_info['email']})! 🎉"
        )

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List all available datasets on the Polaris Hub.

        Args:
            limit: The maximum number of datasets to return.
            offset: The offset from which to start returning datasets.

        Returns:
            A list of dataset names in the format `owner/dataset_name`.
        """
        response = self._base_request_to_hub(
            url="/dataset", method="GET", params={"limit": limit, "offset": offset}
        )
        dataset_list = [bm["artifactId"] for bm in response["data"]]
        return dataset_list

    def get_dataset(self, owner: Union[str, HubOwner], name: str, verify_checksum: bool = True) -> Dataset:
        """Load a dataset from the Polaris Hub.

        Args:
            owner: The owner of the dataset. Can be either a user or organization from the Polaris Hub.
            name: The name of the dataset.
            verify_checksum: Whether to use the checksum to verify the integrity of the dataset.

        Returns:
            A `Dataset` instance, if it exists.
        """

        response = self._base_request_to_hub(url=f"/dataset/{owner}/{name}", method="GET")
        storage_response = self.get(response["tableContent"]["url"])

        # This should be a 307 redirect with the signed URL
        if storage_response.status_code != 307:
            try:
                storage_response.raise_for_status()
            except HTTPStatusError as error:
                raise PolarisHubError("Could not get signed URL from Polaris Hub.") from error

        storage_response = storage_response.json()
        url = storage_response["url"]
        headers = storage_response["headers"]

        response["table"] = self._load_from_signed_url(url=url, headers=headers, load_fn=pd.read_parquet)

        if not verify_checksum:
            response.pop("md5Sum", None)

        return Dataset(**response)

    def open_zarr_file(
        self, owner: Union[str, HubOwner], name: str, path: str, mode: IOMode, as_consolidated: bool = True
    ) -> zarr.hierarchy.Group:
        """Open a Zarr file from a Polaris dataset

        Args:
            owner: Which Hub user or organization owns the artifact.
            name: Name of the dataset.
            path: Path to the Zarr file within the dataset.
            mode: The mode in which the file is opened.
            as_consolidated: Whether to open the store with consolidated metadata for optimized reading. This is only applicable in 'r' and 'r+' modes.

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

        except Exception as e:
            raise PolarisHubError("Error opening Zarr store") from e

    def list_benchmarks(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List all available benchmarks on the Polaris Hub.

        Args:
            limit: The maximum number of benchmarks to return.
            offset: The offset from which to start returning benchmarks.

        Returns:
            A list of benchmark names in the format `owner/benchmark_name`.
        """

        # TODO (cwognum): What to do with pagination, i.e. limit and offset?
        response = self._base_request_to_hub(
            url="/benchmark", method="GET", params={"limit": limit, "offset": offset}
        )
        benchmarks_list = [f"{HubOwner(**bm['owner'])}/{bm['name']}" for bm in response["data"]]
        return benchmarks_list

    def get_benchmark(
        self, owner: Union[str, HubOwner], name: str, verify_checksum: bool = True
    ) -> BenchmarkSpecification:
        """Load a benchmark from the Polaris Hub.

        Args:
            owner: The owner of the benchmark. Can be either a user or organization from the Polaris Hub.
            name: The name of the benchmark.
            verify_checksum: Whether to use the checksum to verify the integrity of the dataset.

        Returns:
            A `BenchmarkSpecification` instance, if it exists.
        """

        response = self._base_request_to_hub(url=f"/benchmark/{owner}/{name}", method="GET")

        # TODO (cwognum): Currently, the benchmark endpoints do not return the owner info for the underlying dataset.
        # TODO (jstlaurent): Use the same owner for now, until the benchmark returns a better dataset entity
        response["dataset"] = self.get_dataset(
            owner, response["dataset"]["name"], verify_checksum=verify_checksum
        )

        # TODO (cwognum): As we get more complicated benchmarks, how do we still find the right subclass?
        #  Maybe through structural pattern matching, introduced in Py3.10, or Pydantic's discriminated unions?
        benchmark_cls = (
            SingleTaskBenchmarkSpecification
            if len(response["targetCols"]) == 1
            else MultiTaskBenchmarkSpecification
        )

        if not verify_checksum:
            response.pop("md5Sum", None)

        return benchmark_cls(**response)

    def upload_results(
        self,
        results: BenchmarkResults,
        access: AccessType = "private",
        owner: Optional[Union[HubOwner, str]] = None,
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

        # Get the serialized model data-structure
        results.owner = self._normalize_owner(results.owner, owner)
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
        logger.success(f"Your result has been successfully uploaded to the Hub. View it here: {result_url}")
        return response

    def upload_dataset(
        self,
        dataset: Dataset,
        access: AccessType = "private",
        timeout: TimeoutTypes = (10, 200),
        owner: Optional[Union[HubOwner, str]] = None,
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
        dataset.owner = self._normalize_owner(dataset.owner, owner)
        dataset_json = dataset.model_dump(exclude={"cache_dir", "table"}, exclude_none=True, by_alias=True)

        # We will save the Zarr archive to the Hub as well
        dataset_json["zarrRootPath"] = f"{PolarisFileSystem.protocol}://data.zarr"

        # Uploading a dataset is a three-step process.
        # 1. Upload the dataset meta data to the hub and prepare the hub to receive the parquet file
        # 2. Upload the parquet file to the hub
        # 3. Upload the associated Zarr archive
        # TODO: Revert step 1 in case step 2 fails - Is this needed? Or should this be taken care of by the hub?

        # Write the parquet file directly to a buffer
        buffer = BytesIO()
        dataset.table.to_parquet(buffer, engine="auto")
        parquet_size = len(buffer.getbuffer())
        parquet_md5 = md5(buffer.getbuffer()).hexdigest()

        # Step 1: Upload meta-data
        # Instead of directly uploading the table, we announce to the hub that we intend to upload one.
        url = f"/dataset/{dataset.artifact_id}"
        response = self._base_request_to_hub(
            url=url,
            method="PUT",
            json={
                "tableContent": {
                    "size": parquet_size,
                    "fileType": "parquet",
                    "md5sum": parquet_md5,
                },
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
                headers={"Content-type": "application/vnd.apache.parquet", **hub_response_body["headers"]},
                content=buffer.getvalue(),
                auth=None,
                timeout=timeout,  # required for large size dataset
            )
            bucket_response.raise_for_status()

        else:
            hub_response.raise_for_status()

        # Step 3: Upload any associated Zarr archive
        if dataset.zarr_root is not None:
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
                    log=logger.info,
                    if_exists=if_exists,
                )

        logger.success(
            "Your dataset has been successfully uploaded to the Hub. "
            f"View it here: {urljoin(self.settings.hub_url, f'datasets/{dataset.owner}/{dataset.name}')}"
        )

        return response

    def upload_benchmark(
        self,
        benchmark: BenchmarkSpecification,
        access: AccessType = "private",
        owner: Optional[Union[HubOwner, str]] = None,
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
        # Get the serialized data-model
        # We exclude the dataset as we expect it to exist on the hub already.
        benchmark.owner = self._normalize_owner(benchmark.owner, owner)
        benchmark_json = benchmark.model_dump(exclude={"dataset"}, exclude_none=True, by_alias=True)
        benchmark_json["datasetArtifactId"] = benchmark.dataset.artifact_id
        benchmark_json["access"] = access

        url = f"/benchmark/{benchmark.owner}/{benchmark.name}"
        response = self._base_request_to_hub(url=url, method="PUT", json=benchmark_json)

        logger.success(
            "Your benchmark has been successfully uploaded to the Hub. "
            f"View it here: {urljoin(self.settings.hub_url, f'benchmarks/{benchmark.owner}/{benchmark.name}')}"
        )
        return response
