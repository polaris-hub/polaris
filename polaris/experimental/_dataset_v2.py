import json
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

import fsspec
import numpy as np
import zarr
from loguru import logger
from pydantic import PrivateAttr, computed_field, model_validator
from typing_extensions import Self

from polaris.dataset._adapters import Adapter
from polaris.dataset._base import BaseDataset
from polaris.dataset.zarr._manifest import calculate_file_md5, generate_zarr_manifest
from polaris.utils.errors import InvalidDatasetError
from polaris.utils.types import AccessType, ChecksumStrategy, HubOwner, ZarrConflictResolution

_INDEX_ARRAY_KEY = "__index__"


class DatasetV2(BaseDataset):
    """Second version of a Polaris Dataset.

    This version gets rid of the DataFrame and stores all data in a Zarr archive.

    V1 stored all datapoints in a Pandas DataFrame. Because a DataFrame is always loaded to memory,
    this was a bottleneck when the number of data points grew large. Even with the pointer columns, you still
    need to load all pointers into memory. V2 therefore switches to a Zarr-only format.

    Info: This feature is still experimental
        The DatasetV2 is in active development and will likely undergo breaking changes before release.

    Attributes:
        zarr_root_path: The path to the Zarr archive. Different from V1, this is now required.

    For additional meta-data attributes, see the [`BaseDataset`][polaris._dataset.BaseDataset] class.

    Raises:
        InvalidDatasetError: If the dataset does not conform to the Pydantic data-model specification.
    """

    _artifact_type = "dataset"

    version: ClassVar[Literal[2]] = 2
    _zarr_manifest_path: str | None = PrivateAttr(None)
    _zarr_manifest_md5sum: str | None = PrivateAttr(None)

    # Redefine this to make it a required field
    zarr_root_path: str

    @model_validator(mode="after")
    def _validate_v2_dataset_model(self) -> Self:
        """Verifies some dependencies between properties"""

        # Since the keys for subgroups are not ordered, we have no easy way to index these groups.
        # Any subgroup should therefore have a special array that defines the index for that group.
        for group in self.zarr_root.group_keys():
            if _INDEX_ARRAY_KEY not in self.zarr_root[group].array_keys():
                raise InvalidDatasetError(f"Group {group} does not have an index array.")

            index_arr = self.zarr_root[group][_INDEX_ARRAY_KEY]
            if len(index_arr) != len(self.zarr_root[group]) - 1:
                raise InvalidDatasetError(
                    f"Length of index array for group {group} does not match the size of the group."
                )
            if any(x not in self.zarr_root[group] for x in index_arr):
                raise InvalidDatasetError(
                    f"Keys of index array for group {group} does not match the group members."
                )

        # Check the structure of the Zarr archive
        # All arrays or groups in the root should have the same length.
        lengths = {len(self.zarr_root[k]) for k in self.zarr_root.array_keys()}
        lengths.update({len(self.zarr_root[k][_INDEX_ARRAY_KEY]) for k in self.zarr_root.group_keys()})
        if len(lengths) > 1:
            raise InvalidDatasetError(
                f"All arrays or groups in the root should have the same length, found the following lengths: {lengths}"
            )

        return self

    @property
    def n_rows(self) -> int:
        """Return all row indices for the dataset"""
        example = self.zarr_root[self.columns[0]]
        if isinstance(example, zarr.Group):
            return len(example[_INDEX_ARRAY_KEY])
        return len(example)

    @property
    def rows(self) -> np.ndarray[int]:
        """Return all row indices for the dataset

        Warning: Memory consumption
            This feature is added for completeness' sake, but it should be noted that large datasets could consume a lot of memory.
            E.g. storing a billion indices with np.in64 would consume 8GB of memory. Use with caution.
        """
        return np.arange(len(self), dtype=int)

    @property
    def columns(self) -> list[str]:
        """Return all columns for the dataset"""
        return list(self.zarr_root.keys())

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Return the dtype for each of the columns for the dataset"""
        dtypes = {}
        for arr in self.zarr_root.array_keys():
            dtypes[arr] = self.zarr_root[arr].dtype
        for group in self.zarr_root.group_keys():
            dtypes[group] = np.dtype(object)
        return dtypes

    def load_zarr_root_from_hub(self):
        """
        Loads a Zarr archive from the Hub.
        """
        from polaris.hub.client import PolarisHubClient
        from polaris.hub.storage import StorageSession

        with PolarisHubClient() as client:
            with StorageSession(client, "read", self.urn) as storage:
                return zarr.open_consolidated(store=storage.root_store)

    @property
    def zarr_manifest_path(self) -> str:
        if self._zarr_manifest_path is None:
            zarr_manifest_path = generate_zarr_manifest(self.zarr_root_path, self._cache_dir)
            self._zarr_manifest_path = zarr_manifest_path

        return self._zarr_manifest_path

    @computed_field
    @property
    def zarr_manifest_md5sum(self) -> str:
        """
        Lazily compute the checksum once needed.

        The checksum of the DatasetV2 is computed from the Zarr Manifest and is _not_ deterministic.
        """
        if not self.has_zarr_manifest_md5sum:
            logger.info("Computing the checksum. This can be slow for large datasets.")
            self.zarr_manifest_md5sum = calculate_file_md5(self.zarr_manifest_path)
        return self._zarr_manifest_md5sum

    @zarr_manifest_md5sum.setter
    def zarr_manifest_md5sum(self, value: str):
        """Set the checksum."""
        if not re.fullmatch(r"^[a-f0-9]{32}$", value):
            raise ValueError("The checksum should be the 32-character hexdigest of a 128 bit MD5 hash.")
        self._zarr_manifest_md5sum = value

    @property
    def has_zarr_manifest_md5sum(self) -> bool:
        """Whether the md5sum for this dataset's zarr manifest has been computed and stored."""
        return self._zarr_manifest_md5sum is not None

    def get_data(self, row: int, col: str, adapters: dict[str, Adapter] | None = None) -> np.ndarray | Any:
        """Indexes the Zarr archive.

        Args:
            row: The index of the data to fetch.
            col: The label of a direct child of the Zarr root.
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

        # Get the data
        group_or_array = self.zarr_data[col]

        # If it is a group, there is no deterministic order for the child keys.
        # We therefore use a special array that defines the index.
        # If loaded to memory, the group is represented by a dictionary.
        if isinstance(group_or_array, zarr.Group) or isinstance(group_or_array, dict):
            # Indices in a group should always be strings
            row = str(group_or_array[_INDEX_ARRAY_KEY][row])
        arr = group_or_array[row]

        # Adapt the input to the specified format
        if adapter is not None:
            arr = adapter(arr)

        return arr

    def upload_to_hub(self, access: AccessType = "private", owner: HubOwner | str | None = None):
        """
        Uploads the dataset to the Polaris Hub.
        """

        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient() as client:
            client.upload_dataset(self, owner=owner, access=access)

    @classmethod
    def from_json(cls, path: str):
        """
        Loads a dataset from a JSON file.

        Args:
            path: The path to the JSON file to load the dataset from.
        """
        with fsspec.open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json(
        self,
        destination: str,
        if_exists: ZarrConflictResolution = "replace",
        load_zarr_from_new_location: bool = False,
    ) -> str:
        """
        Save the dataset to a destination directory as a JSON file.

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

        dataset_path = str(destination / "dataset.json")
        new_zarr_root_path = str(destination / "data.zarr")

        # Lu: Avoid serializing and sending None to hub app.
        serialized = self.model_dump(
            exclude_none=True, exclude={"zarr_manifest_path", "zarr_manifest_md5sum"}
        )
        serialized["zarrRootPath"] = new_zarr_root_path

        # Copy over Zarr data to the destination
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

        with fsspec.open(dataset_path, "w") as f:
            json.dump(serialized, f)
        return dataset_path

    def cache(self) -> str:
        """Caches the dataset by downloading all additional data for pointer columns to a local directory.

        Returns:
            The path to the cache directory.
        """
        self.to_json(self._cache_dir, load_zarr_from_new_location=True)
        return self._cache_dir

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        repr_dict = self.model_dump(exclude={"zarr_manifest_path", "zarr_manifest_md5sum"})
        return repr_dict

    def should_verify_checksum(self, strategy: ChecksumStrategy) -> bool:
        """
        Determines whether to verify the checksum of the dataset based on the strategy.
        """
        return False
