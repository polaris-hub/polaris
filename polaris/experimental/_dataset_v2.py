from typing import ClassVar, List, Literal, Optional

import numpy as np
from pydantic import computed_field

from polaris.dataset._adapters import Adapter
from polaris.dataset._base import BaseDataset
from polaris.dataset.zarr._checksum import ZarrFileChecksum
from polaris.utils.types import AccessType, HubOwner, ZarrConflictResolution


class DatasetV2(BaseDataset):
    """Dataset subclass for Polaris"""

    version: ClassVar[Literal[2]] = 2

    @property
    def rows(self) -> list:
        """Return all row indices for the dataset"""
        raise NotImplementedError

    @property
    def columns(self) -> list:
        """Return all columns for the dataset"""
        raise NotImplementedError

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Return the dtype for each of the columns for the dataset"""
        raise NotImplementedError

    @computed_field
    @property
    def zarr_md5sum_manifest(self) -> List[ZarrFileChecksum]:
        """
        The Zarr Checksum manifest stores the checksums of all files in a Zarr archive.
        If the dataset doesn't use Zarr, this will simply return an empty list.
        """
        raise NotImplementedError

    def _compute_checksum(self) -> str:
        """Compute the checksum of the dataset."""
        raise NotImplementedError

    def get_data(self, row: str | int, col: str, adapters: List[Adapter] | None = None) -> np.ndarray:
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
        raise NotImplementedError

    def upload_to_hub(self, access: Optional[AccessType] = "private", owner: HubOwner | str | None = None):
        """Uploads the dataset to the Polaris Hub."""
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
        raise NotImplementedError

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        raise NotImplementedError
