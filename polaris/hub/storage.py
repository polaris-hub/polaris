from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from hashlib import md5
from io import BytesIO
from pathlib import Path
from typing import Any, Generator, Literal, Mapping, Sequence, TypeAlias

import boto3
from authlib.integrations.httpx_client import OAuth2Client, OAuthError
from authlib.oauth2 import OAuth2Error
from authlib.oauth2.rfc6749 import OAuth2Token
from botocore.exceptions import BotoCoreError, ClientError
from typing_extensions import Self
from zarr.context import Context
from zarr.storage import Store

from polaris.hub.oauth import DatasetV1Paths, DatasetV2Paths, HubStorageOAuth2Token
from polaris.utils.errors import PolarisHubError
from polaris.utils.types import ArtifactUrn

Scope: TypeAlias = Literal["read", "write"]


class S3StoreException(Exception):
    """
    Base exception for S3Store.
    """


class S3StoreCredentialsExpiredException(S3StoreException):
    """
    Exception raised when the S3 credentials have expired.
    """


@contextmanager
def handle_s3_errors():
    """
    Standardize error handling for S3 operations.
    """
    try:
        yield
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ExpiredToken":
            raise S3StoreCredentialsExpiredException(f"Error in S3Store: Credentials expired: {e}") from e
        else:
            raise S3StoreException(f"Error in S3Store: {e}") from e
    except BotoCoreError as e:
        raise S3StoreException(f"Error in S3Store: {e}") from e


class S3Store(Store):
    """
     A Zarr store implementation using a S3 bucket as the backend storage.

    It supports multipart uploads for large objects and handles S3-specific exceptions.
    """

    _erasable = False

    def __init__(
        self,
        path: str,
        access_key: str,
        secret_key: str,
        token: str,
        endpoint_url: str,
        part_size: int = 10 * 1024 * 1024,  # 10MB
        content_type: str = "application/octet-stream",
    ) -> None:
        bucket_name, prefix = path.split("/", 1)

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=token,
            endpoint_url=endpoint_url,
        )
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.part_size = part_size
        self.content_type = content_type

    def _full_key(self, key: str) -> str:
        """
        Converts a relative key to the full bucket key.
        """
        return f"{self.prefix}/{key}"

    def _multipart_upload(self, key: str, value: bytes) -> None:
        """
        For large files, use multipart to split the work.
        """
        full_key = self._full_key(key)
        md5_hash = md5(value)
        with handle_s3_errors():
            upload = self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=full_key,
                ContentType=self.content_type,
                Metadata={
                    "md5sum": md5_hash.hexdigest(),
                },
            )
            upload_id = upload["UploadId"]

            parts = []
            for i in range(0, len(value), self.part_size):
                part_number = i // self.part_size + 1
                part = value[i : i + self.part_size]
                response = self.s3_client.upload_part(
                    Bucket=self.bucket_name,
                    Key=full_key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=part,
                    ContentMD5=b64encode(md5(part).digest()).decode(),
                )
                parts.append({"ETag": response["ETag"], "PartNumber": part_number})

            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name, Key=full_key, UploadId=upload_id, MultipartUpload={"Parts": parts}
            )

    def listdir(self, path: str = "") -> Generator[str, None, None]:
        """
        For a given path, list all the "subdirectories" and "files" for that path.
        The returned paths are relative to the input path.

        Uses pagination and return a generator to handle very large number of keys.
        Note: This might not help with some Zarr operations that materialize the whole sequence.
        """
        prefix = self._full_key(path)

        # Ensure a trailing slash to avoid the path looking like one specific key
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        with handle_s3_errors():
            # `list_objects_v2` returns a max of 1000 keys per request, so paginate requests
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter="/")

            for page in page_iterator:
                # Contents are "files"
                for obj in page.get("Contents", []):
                    key = obj["Key"][len(prefix) :]
                    if key:
                        yield key.split("/")[0]

                # CommonPrefixes are "subdirectories"
                for common_prefix in page.get("CommonPrefixes", []):
                    yield common_prefix["Prefix"][len(prefix) :].strip("/")

    def getitems(self, keys: Sequence[str], *, contexts: Mapping[str, Context]) -> dict[str, Any]:
        """
        More efficient implementation of getitems using concurrent fetching through multiple threads.

        The default implementation uses __contains__ to check existence before fetching the value,
        which doubles the number of requests.
        """

        def fetch_key(key: str) -> tuple[str, Any]:
            """
            Wrapper that looks up a key's value in the store
            """
            try:
                return key, self[key]
            except KeyError:
                return key, None

        results = {}
        with ThreadPoolExecutor() as executor:
            # Create a future for each key to look up
            future_to_key = {executor.submit(fetch_key, key): key for key in keys}

            # As each future completes, collect the results, if any
            for future in as_completed(future_to_key):
                key, value = future.result()
                if value is not None:
                    results[key] = value

        return results

    def getsize(self, key: str) -> int:
        """
        Return the size (in bytes) of the object at the given key.
        """
        with handle_s3_errors():
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=self._full_key(key))
            return response["ContentLength"]

    ## MutableMapping implementation, expected by Zarr

    def __getitem__(self, key: str) -> bytes:
        """
        Retrieves the value for the given key from the store.

        Makes no provision to handle overly large values returned.
        """
        with handle_s3_errors():
            try:
                full_key = self._full_key(key)
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_key)
                return response["Body"].read()
            except self.s3_client.exceptions.NoSuchKey:
                raise KeyError(key)

    def __setitem__(self, key: str, value: bytes | bytearray | memoryview) -> None:
        """
        Persists the given value in the store.

        Based on value size, will use multipart upload for large files,
        or a single put_object call.
        """
        if isinstance(value, memoryview):
            value = value.tobytes()

        if len(value) > self.part_size:
            self._multipart_upload(key, value)
        else:
            with handle_s3_errors():
                md5_hash = md5(value)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=self._full_key(key),
                    Body=value,
                    ContentType=self.content_type,
                    ContentMD5=b64encode(md5_hash.digest()).decode(),
                    Metadata={
                        "md5sum": md5_hash.hexdigest(),
                    },
                )

    def __delitem__(self, key: str) -> None:
        """
        Removing a key from the store is not supported.
        """
        raise NotImplementedError(f'{type(self)} is not erasable, cannot call "del store[key]"')

    def __contains__(self, key: str) -> bool:
        """
        Checks the existence of a key in the store.

        If the intent is to download the value after this check, it is more efficient to
        attempt tp retrieve it and handle the KeyError from a non-existent key.
        """
        with handle_s3_errors():
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=self._full_key(key))
                return True
            except self.s3_client.exceptions.NoSuchKey:
                return False
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False
                raise e

    def __iter__(self) -> Generator[str, None, None]:
        """
        Iterate through all the keys in the store.
        """
        with handle_s3_errors():
            # `list_objects_v2` returns a max of 1000 keys per request, so paginate requests
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

            for page in page_iterator:
                for obj in page.get("Contents", []):
                    yield obj["Key"][len(self.prefix) + 1 :]

    def __len__(self) -> int:
        """
        Number of keys in the store.
        """
        with handle_s3_errors():
            # `list_objects_v2` returns a max of 1000 keys per request, so paginate requests
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

            return sum((page["KeyCount"] for page in page_iterator))


class StorageTokenAuth:
    token: HubStorageOAuth2Token | None

    def __init__(self, token: dict[str, Any] | None, *args) -> None:
        self.token = None
        if token:
            self.set_token(token)

    def set_token(self, token: dict[str, Any] | HubStorageOAuth2Token) -> None:
        self.token = HubStorageOAuth2Token(**token) if isinstance(token, dict) else token


class StorageSession(OAuth2Client):
    """
    A context manager for managing a storage session, with token exchange and token refresh capabilities.
    Each session is associated with a specific scope and resource.
    """

    polaris_protocol = "polarisfs"

    token_auth_class = StorageTokenAuth

    def __init__(self, hub_client, scope: Scope, resource: ArtifactUrn):
        self.hub_client = hub_client
        self.resource = resource

        super().__init__(
            # OAuth2Client
            token_endpoint=hub_client.settings.hub_token_url,
            token_endpoint_auth_method="none",
            grant_type="urn:ietf:params:oauth:grant-type:token-exchange",
            scope=scope,
            # httpx.Client
            cert=hub_client.settings.ca_bundle,
        )

    def __enter__(self) -> Self:
        self.ensure_active_token()
        return self

    def _prepare_token_endpoint_body(self, body, grant_type, **kwargs) -> str:
        """
        Override to support required fields for the token exchange grant type.
        See https://datatracker.ietf.org/doc/html/rfc8693#name-request
        """
        if grant_type == "urn:ietf:params:oauth:grant-type:token-exchange":
            kwargs.update(
                {
                    "subject_token": self.hub_client.token.get("access_token"),
                    "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                    "requested_token_type": "urn:ietf:params:oauth:token-type:jwt",
                    "resource": self.resource,
                }
            )
        return super()._prepare_token_endpoint_body(body, grant_type, **kwargs)

    def fetch_token(self, **kwargs) -> dict[str, Any]:
        """
        Error handling for token fetching.
        """
        try:
            return super().fetch_token(**kwargs)
        except (OAuthError, OAuth2Error) as error:
            raise PolarisHubError(
                message=f"Could not obtain a token to access the storage backend. Error was: {error.error} - {error.description}"
            ) from error

    def ensure_active_token(self, token: OAuth2Token | None = None) -> bool:
        """
        Override the active check to trigger a re-fetch of the token if it is not active.
        """
        if token is None:
            # This won't be needed with if we set a lower bound for authlib: >=1.3.2
            # See https://github.com/lepture/authlib/pull/625
            # As of now, this latest version is not available on Conda though.
            token = self.token

        if token and super().ensure_active_token(token):
            return True

        # Check if external token is still valid
        if not self.hub_client.ensure_active_token():
            return False

        # If so, use it to get a new Hub token
        self.token = self.fetch_token()
        return True

    @property
    def paths(self) -> DatasetV1Paths | DatasetV2Paths:
        return self.token.extra_data.paths

    def _set_file(self, path: Path, value: bytes | bytearray) -> None:
        """
        Internal method to upload non-zarr file to the storage backend.
        """
        storage_data = self.token.extra_data

        match path.suffix:
            case ".parquet":
                content_type = "application/vnd.apache.parquet"
            case _:
                content_type = "application/octet-stream"

        store = S3Store(
            path=str(path.parent),
            access_key=storage_data.key,
            secret_key=storage_data.secret,
            token=f"jwt/{self.token.access_token}",
            endpoint_url=storage_data.endpoint,
            content_type=content_type,
        )
        store[path.name] = value

    def _get_file(self, path: Path) -> BytesIO:
        """
        Internal method to download non-zarr file from the storage backend.
        """
        storage_data = self.token.extra_data

        store = S3Store(
            path=str(path.parent),
            access_key=storage_data.key,
            secret_key=storage_data.secret,
            token=f"jwt/{self.token.access_token}",
            endpoint_url=storage_data.endpoint,
        )

        return BytesIO(store[path.name])

    def set_root(self, value: bytes | bytearray) -> None:
        """
        Set a value at the root path.
        """
        if not isinstance(self.paths, DatasetV1Paths):
            raise NotImplementedError("Only DatasetV1Paths are supported for setting the root path")

        path = Path(self.paths.relative_root)
        self._set_file(path, value)

    def get_root(self) -> BytesIO:
        """
        Get the value at the root path.
        """
        if not isinstance(self.paths, DatasetV1Paths):
            raise NotImplementedError("Only DatasetV1Paths are supported for getting the root path")

        path = Path(self.paths.relative_root)
        return self._get_file(path)

    def set_manifest(self, value: bytes | bytearray) -> None:
        """
        Set a value at the manifest path.
        """
        if not isinstance(self.paths, DatasetV2Paths):
            raise NotImplementedError("Only DatasetV2Paths are supported for setting the manifest path")

        path = Path(self.paths.relative_manifest)
        self._set_file(path, value)

    def get_manifest(self) -> BytesIO:
        """
        Get the value at the manifest path.
        """
        if not isinstance(self.paths, DatasetV2Paths):
            raise NotImplementedError("Only DatasetV2Paths are supported for getting the manifest path")

        path = Path(self.paths.relative_manifest)
        return self._get_file(path)

    @property
    def extension_store(self) -> S3Store | None:
        """
        Returns a Zarr store for the extension path, if available, backed by a S3 compatible bucket.
        """
        storage_data = self.token.extra_data
        match storage_data.paths:
            case DatasetV1Paths() if storage_data.paths.relative_extension:
                return S3Store(
                    path=storage_data.paths.relative_extension,
                    access_key=storage_data.key,
                    secret_key=storage_data.secret,
                    token=f"jwt/{self.token.access_token}",
                    endpoint_url=storage_data.endpoint,
                )
            case _:
                return None

    @property
    def root_store(self) -> S3Store | None:
        """
        Returns a Zarr store for the root path, backed by a S3 compatible bucket.
        """
        storage_data = self.token.extra_data

        match storage_data.paths:
            case DatasetV2Paths():
                return S3Store(
                    path=storage_data.paths.relative_root,
                    access_key=storage_data.key,
                    secret_key=storage_data.secret,
                    token=f"jwt/{self.token.access_token}",
                    endpoint_url=storage_data.endpoint,
                )
            case _:
                return None
