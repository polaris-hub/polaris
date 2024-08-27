import json
from pathlib import Path
from typing import ClassVar, List, Literal, Optional

import fsspec
import numpy as np
import zarr
from loguru import logger
from pydantic import computed_field, model_validator

from polaris.dataset._adapters import Adapter
from polaris.dataset._base import BaseDataset
from polaris.dataset._column import ColumnAnnotation
from polaris.dataset.zarr._checksum import ZarrFileChecksum, compute_zarr_checksum
from polaris.utils.errors import InvalidDatasetError
from polaris.utils.types import AccessType, HubOwner, ZarrConflictResolution

_INDEX_ARRAY_KEY = "__index__"


class DatasetV2(BaseDataset):
    """Dataset subclass for Polaris"""

    version: ClassVar[Literal[2]] = 2

    # Redefine this to make it a required field
    zarr_root_path: str

    @model_validator(mode="after")
    @classmethod
    def _validate_model(cls, m: "DatasetV2"):
        """Verifies some dependencies between properties"""

        # NOTE (cwognum): A good chunk of the below code is shared with the DatasetV1 class.
        #  I tried moving it to the BaseDataset class, but I'm not understanding Pydantic's behavior very well.
        #  It seems to not always trigger when part of the base class.

        # Verify that all annotations are for columns that exist
        if any(k not in m.columns for k in m.annotations):
            raise InvalidDatasetError(
                f"There are annotations for columns that do not exist. Columns: {m.columns}. Annotations: {list(m.annotations.keys())}"
            )

        # Verify that all adapters are for columns that exist
        if any(k not in m.columns for k in m.default_adapters.keys()):
            raise InvalidDatasetError(
                f"There are default adapters for columns that do not exist. Columns: {m.columns}. Adapters: {list(m.annotations.keys())}"
            )

        # Set a default for missing annotations and convert strings to Modality
        for c in m.columns:
            if c not in m.annotations:
                m.annotations[c] = ColumnAnnotation()
            if m.annotations[c].is_pointer:
                raise InvalidDatasetError("Pointer columns are not supported in DatasetV2")
            m.annotations[c].dtype = m.dtypes[c]

        # Since the keys for subgroups are not ordered, we have no easy way to index these groups.
        # Any subgroup should therefore have a special array that defines the index for that group.
        for group in m.zarr_root.group_keys():
            if _INDEX_ARRAY_KEY not in m.zarr_root[group].array_keys():
                raise InvalidDatasetError(f"Group {group} does not have an index array.")

            index_arr = m.zarr_root[group][_INDEX_ARRAY_KEY]
            if len(index_arr) != len(m.zarr_root[group]) - 1:
                raise InvalidDatasetError(
                    f"Length of index array for group {group} does not match the size of the group."
                )
            if any(x not in m.zarr_root[group] for x in index_arr):
                raise InvalidDatasetError(
                    f"Keys of index array for group {group} does not match the group members."
                )

        # Check the structure of the Zarr archive
        # All arrays or groups in the root should have the same length.
        lengths = {len(m.zarr_root[k]) for k in m.zarr_root.array_keys()}
        lengths.update({len(m.zarr_root[k][_INDEX_ARRAY_KEY]) for k in m.zarr_root.group_keys()})
        if len(lengths) > 1:
            raise InvalidDatasetError(
                f"All arrays or groups in the root should have the same length, found the following lengths: {lengths}"
            )
        return m

    @property
    def n_rows(self) -> list:
        """Return all row indices for the dataset"""
        example = self.zarr_root[self.columns[0]]
        if isinstance(example, zarr.Group):
            return len(example[_INDEX_ARRAY_KEY])
        return len(example)

    @property
    def rows(self) -> list:
        """Return all row indices for the dataset"""
        return np.arange(len(self))

    @property
    def columns(self) -> list:
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

    def _compute_checksum(self) -> str:
        """Compute the checksum of the dataset."""
        zarr_hash, self._zarr_md5sum_manifest = compute_zarr_checksum(self.zarr_root_path)
        return zarr_hash.md5

    def get_data(self, row: int, col: str, adapters: List[Adapter] | None = None) -> np.ndarray:
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
        adapters = adapters or self.default_adapters
        adapter = adapters.get(col)

        # Get the data
        group_or_array = self.zarr_data[col]

        if isinstance(group_or_array, zarr.Group):
            row = group_or_array[_INDEX_ARRAY_KEY][row]
        arr = group_or_array[row]

        # Adapt the input to the specified format
        if adapter is not None:
            arr = adapter(arr)

        return arr

    def upload_to_hub(self, access: Optional[AccessType] = "private", owner: HubOwner | str | None = None):
        """Uploads the dataset to the Polaris Hub."""

        # NOTE (cwognum):  Leaving this for a later PR, because I want
        #  to do it simultaneously with a PR on the Hub side.
        raise NotImplementedError

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

        # Lu: Avoid serilizing and sending None to hub app.
        serialized = self.model_dump(exclude={"cache_dir"}, exclude_none=True)
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

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        repr_dict = self.model_dump(exclude={"zarr_md5sum_manifest"})
        return repr_dict
