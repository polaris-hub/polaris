import json
import uuid
from hashlib import md5
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Union

import fsspec
import numpy as np
import pandas as pd
import zarr
from datamol.utils import fs as dmfs
from loguru import logger
from pydantic import PrivateAttr, computed_field, field_validator, model_validator

from polaris.dataset._adapters import Adapter
from polaris.dataset._base import _CACHE_SUBDIR, BaseDataset
from polaris.dataset.zarr import ZarrFileChecksum, compute_zarr_checksum
from polaris.mixins._checksum import ChecksumMixin
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.errors import InvalidDatasetError
from polaris.utils.types import (
    AccessType,
    HubOwner,
    ZarrConflictResolution,
    TimeoutTypes,
)

# Constants
_SUPPORTED_TABLE_EXTENSIONS = ["parquet"]
_INDEX_SEP = "#"


class DatasetV1(BaseDataset, ChecksumMixin):
    """First version of a Polaris Dataset.

    Stores datapoints in a Pandas DataFrame and implements _pointer columns_ to support the storage of XXL data
    outside of the DataFrame in a Zarr archive.

    Info: Pointer columns
        For complex data, such as images, we support storing the content in external blobs of data.
        In that case, the table contains _pointers_ to these blobs that are dynamically loaded when needed.

    Attributes:
        table: The core data-structure, storing data-points in a row-wise manner. Can be specified as either a
            path to a `.parquet` file or a `pandas.DataFrame`.

    For additional meta-data attributes, see the [`BaseDataset`][polaris.dataset._base.BaseDataset] class.

    Raises:
        InvalidDatasetError: If the dataset does not conform to the Pydantic data-model specification.
    """

    # Public attributes
    # Data
    table: pd.DataFrame

    version: ClassVar[Literal[1]] = 1
    _zarr_md5sum_manifest: List[ZarrFileChecksum] = PrivateAttr(default_factory=list)

    @field_validator("table", mode="before")
    def _validate_table(cls, v):
        """
        If the table is not a dataframe yet, assume it's a path and try load it.
        We also make sure that the pandas index is contiguous and starts at 0, and
        that all columns are named and unique.
        """
        # Load from path if not a dataframe
        if not isinstance(v, pd.DataFrame):
            if not dmfs.is_file(v) or dmfs.get_extension(v) not in _SUPPORTED_TABLE_EXTENSIONS:
                raise InvalidDatasetError(f"{v} is not a valid DataFrame or .parquet path.")
            v = pd.read_parquet(v)
        # Check if there are any duplicate columns
        if any(v.columns.duplicated()):
            raise InvalidDatasetError("The table contains duplicate columns")
        # Check if there are any unnamed columns
        if not all(isinstance(c, str) for c in v.columns):
            raise InvalidDatasetError("The table contains unnamed columns")
        return v

    @model_validator(mode="after")
    @classmethod
    def _validate_v1_dataset_model(cls, m: "DatasetV1"):
        """Verifies some dependencies between properties"""

        has_pointers = any(anno.is_pointer for anno in m.annotations.values())
        if has_pointers and m.zarr_root_path is None:
            raise InvalidDatasetError("A zarr_root_path needs to be specified when there are pointer columns")
        if not has_pointers and m.zarr_root_path is not None:
            raise InvalidDatasetError(
                "The zarr_root_path should only be specified when there are pointer columns"
            )

        # Set the default cache dir if none and make sure it exists
        if m.cache_dir is None:
            dataset_id = m._md5sum if m.has_md5sum else str(uuid.uuid4())
            m.cache_dir = Path(DEFAULT_CACHE_DIR) / _CACHE_SUBDIR / dataset_id
        m.cache_dir.mkdir(parents=True, exist_ok=True)

        return m

    def _compute_checksum(self) -> str:
        """Computes a hash of the dataset.

        This is meant to uniquely identify the dataset and can be used to verify the version.

        1. Is not sensitive to the ordering of the columns or rows in the table.
        2. Purposefully does not include the meta-data (source, description, name, annotations).
        3. Includes a hash for the Zarr archive.
        """
        hash_fn = md5()

        # Sort the columns s.t. the checksum is not sensitive to the column-ordering
        df = self.table.copy(deep=True)
        df = df[sorted(df.columns.tolist())]

        # Use the sum of the row-wise hashes s.t. the hash is insensitive to the row-ordering
        table_hash = pd.util.hash_pandas_object(df, index=False).sum()
        hash_fn.update(table_hash)

        # If the Zarr archive exists, we hash its contents too.
        if self.uses_zarr:
            zarr_hash, self._zarr_md5sum_manifest = compute_zarr_checksum(self.zarr_root_path)
            hash_fn.update(zarr_hash.digest.encode())

        checksum = hash_fn.hexdigest()
        return checksum

    @computed_field
    @property
    def zarr_md5sum_manifest(self) -> List[ZarrFileChecksum]:
        """
        The Zarr Checksum manifest stores the checksums of all files in a Zarr archive.
        If the dataset doesn't use Zarr, this will simply return an empty list.
        """
        if len(self._zarr_md5sum_manifest) == 0 and not self.has_md5sum:
            # The manifest is set as an instance variable
            # as a side-effect of the compute_checksum method
            self.md5sum = self._compute_checksum()
        return self._zarr_md5sum_manifest

    @property
    def rows(self) -> list[str | int]:
        """Return all row indices for the dataset"""
        return self.table.index.tolist()

    @property
    def columns(self) -> list[str]:
        """Return all columns for the dataset"""
        return self.table.columns.tolist()

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Return the dtype for each of the columns for the dataset"""
        return {col: self.table[col].dtype for col in self.columns}

    def get_data(
        self, row: str | int, col: str, adapters: dict[str, Adapter] | None = None
    ) -> np.ndarray | Any:
        """Since the dataset might contain pointers to external files, data retrieval is more complicated
        than just indexing the `table` attribute. This method provides an end-point for seamlessly
        accessing the underlying data.

        Args:
            row: The row index in the `Dataset.table` attribute
            col: The column index in the `Dataset.table` attribute
            adapters: The adapters to apply to the data before returning it.
                If None, will use the default adapters specified for the dataset.

        Returns:
            A numpy array with the data at the specified indices. If the column is a pointer column,
                the content of the referenced file is loaded to memory.
        """

        # Fetch adapters for dataset and a given column
        # Partially override if the adapters parameter is specified.
        adapters = {**self.default_adapters, **(adapters or {})}
        adapter = adapters.get(col)

        # If not a pointer, return it here. Apply adapter if specified.
        value = self.table.at[row, col]
        if not self.annotations[col].is_pointer:
            if adapter is not None:
                return adapter(value)
            return value

        # Load the data from the Zarr archive
        path, index = self._split_index_from_path(value)
        arr = self.zarr_data[path][index]

        # Change to tuple if a slice
        if isinstance(index, slice):
            arr = tuple(arr)

        # Adapt the input to the specified format
        if adapter is not None:
            arr = adapter(arr)

        return arr

    def upload_to_hub(
        self,
        access: Optional[AccessType] = "private",
        owner: Union[HubOwner, str, None] = None,
        timeout: TimeoutTypes = (10, 200),
    ):
        """
        Very light, convenient wrapper around the
        [`PolarisHubClient.upload_dataset`][polaris.hub.client.PolarisHubClient.upload_dataset] method.
        """
        self.client.upload_dataset(self, access=access, owner=owner, timeout=timeout)

    def to_json(
        self,
        destination: str,
        if_exists: ZarrConflictResolution = "replace",
        load_zarr_from_new_location: bool = False,
    ) -> str:
        """
        Save the dataset to a destination directory as a JSON file.

        Warning: Multiple files
            Perhaps unintuitive, this method creates multiple files.

            1. `/path/to/destination/dataset.json`: This file can be loaded with `Dataset.from_json`.
            2. `/path/to/destination/table.parquet`: The `Dataset.table` attribute is saved here.
            3. _(Optional)_ `/path/to/destination/data/*`: Any additional blobs of data referenced by the
                    pointer columns will be stored here.

        Args:
            destination: The _directory_ to save the associated data to.
            if_exists: Action for handling existing files in the Zarr archive. Options are 'raise' to throw
                an error, 'replace' to overwrite, or 'skip' to proceed without altering the existing files.
            load_zarr_from_new_location: Whether to update the current instance to load data from the location
                the data is saved to. Only relevant for Zarr-datasets.

        Returns:
            The path to the JSON file.
        """
        destination = Path(destination)
        destination.mkdir(exist_ok=True, parents=True)

        table_path = str(destination / "table.parquet")
        dataset_path = str(destination / "dataset.json")
        new_zarr_root_path = str(destination / "data.zarr")

        # Lu: Avoid serilizing and sending None to hub app.
        serialized = self.model_dump(exclude={"cache_dir"}, exclude_none=True)
        serialized["table"] = table_path

        # Copy over Zarr data to the destination
        if self.uses_zarr:
            serialized["zarrRootPath"] = new_zarr_root_path

            self._warn_about_remote_zarr = False

            logger.info(f"Copying Zarr archive to {new_zarr_root_path}. This may take a while.")

            dest = zarr.open(new_zarr_root_path, "w")

            zarr.copy_store(
                source=self.zarr_root.store.store,
                dest=dest.store,
                log=logger.debug,
                if_exists=if_exists,
            )

            if load_zarr_from_new_location:
                self.zarr_root_path = new_zarr_root_path
                self._zarr_root = None
                self._zarr_data = None

        self.table.to_parquet(table_path)
        with fsspec.open(dataset_path, "w") as f:
            json.dump(serialized, f)

        return dataset_path

    def cache(self, verify_checksum: bool = False):
        """Cache the dataset to the cache directory.

        Args:
            verify_checksum: Whether to verify the checksum of the dataset after caching.
        """
        dst = super().cache()
        if verify_checksum:
            self.verify_checksum()
        return dst

    def _split_index_from_path(self, path: str) -> tuple[str, int | None]:
        """
        Paths can have an additional index appended to them.
        This extracts that index from the path.
        """
        index = None
        if _INDEX_SEP in path:
            path, index = path.split(_INDEX_SEP)
            index = index.split(":")

            if len(index) == 1:
                index = int(index[0])
            elif len(index) == 2:
                index = slice(int(index[0]), int(index[1]))
            else:
                raise ValueError(f"Invalid index format: {index}")
        return path, index

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        repr_dict = self.model_dump(exclude={"table", "zarr_md5sum_manifest"})
        return repr_dict

    def __eq__(self, other):
        """Whether two datasets are equal is solely determined by the checksum"""
        if not isinstance(other, DatasetV1):
            return False
        return self.md5sum == other.md5sum
