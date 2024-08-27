import abc
import json
import uuid
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Union

import fsspec
import numpy as np
import pandas as pd
import zarr
from loguru import logger
from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from polaris._artifact import BaseArtifactModel
from polaris.dataset._adapters import Adapter
from polaris.dataset._column import ColumnAnnotation
from polaris.dataset.zarr import MemoryMappedDirectoryStore, ZarrFileChecksum
from polaris.dataset.zarr._utils import load_zarr_group_to_memory
from polaris.hub.polarisfs import PolarisFileSystem
from polaris.mixins import ChecksumMixin
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidDatasetError
from polaris.utils.types import (
    AccessType,
    HttpUrlString,
    HubOwner,
    SupportedLicenseType,
    ZarrConflictResolution,
)

# Constants
_CACHE_SUBDIR = "datasets"


class BaseDataset(BaseArtifactModel, ChecksumMixin, abc.ABC):
    """Base data-model for a Polaris dataset, implemented as a [Pydantic](https://docs.pydantic.dev/latest/) model.

    At its core, a dataset in Polaris is a tabular data structure that stores data-points in a row-wise manner.
    A Dataset can have multiple modalities or targets, can be sparse and can be part of one or multiple
     [`BenchmarkSpecification`][polaris.benchmark.BenchmarkSpecification] objects.

    Info: Pointer columns
        Whereas a `Dataset` contains all information required to construct a dataset, it is not ready yet.
        For complex data, such as images, we support storing the content in external blobs of data.
        In that case, the table contains _pointers_ to these blobs that are dynamically loaded when needed.

    Attributes:
        default_adapters: The adapters that the Dataset recommends to use by default to change the format of the data
            for specific columns.
        zarr_root_path: The data for any pointer column should be saved in the Zarr archive this path points to.
        readme: Markdown text that can be used to provide a formatted description of the dataset.
            If using the Polaris Hub, it is worth noting that this field is more easily edited through the Hub UI
            as it provides a rich text editor for writing markdown.
        annotations: Each column _can be_ annotated with a [`ColumnAnnotation`][polaris.dataset.ColumnAnnotation] object.
            Importantly, this is used to annotate whether a column is a pointer column.
        source: The data source, e.g. a DOI, Github repo or URI.
        license: The dataset license. Polaris only supports some Creative Commons licenses. See [`SupportedLicenseType`][polaris.utils.types.SupportedLicenseType] for accepted ID values.
        curation_reference: A reference to the curation process, e.g. a DOI, Github repo or URI.
    For additional meta-data attributes, see the [`BaseArtifactModel`][polaris._artifact.BaseArtifactModel] class.

    Raises:
        InvalidDatasetError: If the dataset does not conform to the Pydantic data-model specification.
    """

    # Public attributes
    # Data
    default_adapters: Dict[str, Adapter] = Field(default_factory=dict)
    zarr_root_path: Optional[str] = None

    # Additional meta-data
    readme: str = ""
    annotations: Dict[str, ColumnAnnotation] = Field(default_factory=dict)
    source: Optional[HttpUrlString] = None
    license: Optional[SupportedLicenseType] = None
    curation_reference: Optional[HttpUrlString] = None

    # Config
    cache_dir: Optional[Path] = None  # Where to cache the data to if cache() is called.

    # Private attributes
    _zarr_root: Optional[zarr.Group] = PrivateAttr(None)
    _zarr_data: Optional[MutableMapping[str, np.ndarray]] = PrivateAttr(None)
    _zarr_md5sum_manifest: List[ZarrFileChecksum] = PrivateAttr(default_factory=list)
    _client = PrivateAttr(None)  # Optional[PolarisHubClient]
    _warn_about_remote_zarr: bool = PrivateAttr(True)

    @model_validator(mode="after")
    @classmethod
    def _validate_model(cls, m: "BaseDataset"):
        """Verifies some dependencies between properties"""

        # Set the default cache dir if none and make sure it exists
        if m.cache_dir is None:
            dataset_id = m._md5sum if m.has_md5sum else str(uuid.uuid4())
            m.cache_dir = Path(DEFAULT_CACHE_DIR) / _CACHE_SUBDIR / dataset_id

        m.cache_dir.mkdir(parents=True, exist_ok=True)
        return m

    @field_validator("default_adapters", mode="before")
    def _validate_adapters(cls, value):
        """Validate the adapters"""
        return {k: Adapter[v] if isinstance(v, str) else v for k, v in value.items()}

    @field_serializer("default_adapters")
    def _serialize_adapters(self, value: List[Adapter]):
        """Serializes the adapters"""
        return {k: v.name for k, v in value.items()}

    @field_serializer("cache_dir", "zarr_root_path")
    def _serialize_paths(value):
        """Serialize the paths"""
        if value is not None:
            value = str(value)
        return value

    @computed_field
    @property
    @abc.abstractmethod
    def zarr_md5sum_manifest(self) -> List[ZarrFileChecksum]:
        """
        The Zarr Checksum manifest stores the checksums of all files in a Zarr archive.
        If the dataset doesn't use Zarr, this will simply return an empty list.
        """
        raise NotImplementedError

    @property
    def client(self):
        """The Polaris Hub client used to interact with the Polaris Hub."""

        # Import it here to prevent circular imports
        from polaris.hub.client import PolarisHubClient

        if self._client is None:
            self._client = PolarisHubClient()
        return self._client

    @property
    def uses_zarr(self) -> bool:
        """Whether any of the data in this dataset is stored in a Zarr Archive."""
        return self.zarr_root_path is not None

    @property
    def zarr_data(self):
        """Get the Zarr data.

        This is different from the Zarr Root, because to optimize the efficiency of
        data loading, a user can choose to load the data into memory as a numpy array

        Note: General purpose dataloader.
            The goal with Polaris is to provide general purpose datasets that serve as good
            options for a _wide variety of use cases_. This also implies you should be able to
            optimize things further for a specific use case if needed.
        """
        if self._zarr_data is not None:
            return self._zarr_data
        return self.zarr_root

    @property
    def zarr_root(self):
        """Get the zarr Group object corresponding to the root.

        Opens the zarr archive in read-write mode if it is not already open.

        Note: Different to `zarr_data`
            The `zarr_data` attribute references either to the Zarr archive or to a in-memory copy of the data.
            See also [`Dataset.load_to_memory`][polaris.dataset.Dataset.load_to_memory].
        """

        if self._zarr_root is not None:
            return self._zarr_root

        if self.zarr_root_path is None:
            return None

        # We open the archive in read-only mode if it is saved on the Hub
        saved_on_hub = PolarisFileSystem.is_polarisfs_path(self.zarr_root_path)

        if self._warn_about_remote_zarr:
            saved_remote = saved_on_hub or not Path(self.zarr_root_path).exists()

            if saved_remote:
                logger.warning(
                    f"You're loading data from a remote location. "
                    f"To speed up this process, consider caching the dataset first "
                    f"using {self.__class__.__name__}.cache()"
                )
                self._warn_about_remote_zarr = False

        try:
            if saved_on_hub:
                self._zarr_root = self.client.open_zarr_file(self.owner, self.name, self.zarr_root_path, "r+")
            else:
                # We use memory mapping by default because our experiments show that it's consistently faster
                store = MemoryMappedDirectoryStore(self.zarr_root_path)
                self._zarr_root = zarr.open_consolidated(store, mode="r+")
        except KeyError as error:
            raise InvalidDatasetError(
                "A Zarr archive associated with a Polaris dataset has to be consolidated."
            ) from error
        return self._zarr_root

    @computed_field
    @property
    def n_rows(self) -> int:
        """The number of rows in the dataset."""
        return len(self.rows)

    @computed_field
    @property
    def n_columns(self) -> int:
        """The number of columns in the dataset."""
        return len(self.columns)

    @property
    @abc.abstractmethod
    def rows(self) -> list:
        """Return all row indices for the dataset"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def columns(self) -> list:
        """Return all columns for the dataset"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtypes(self) -> dict[str, np.dtype]:
        """Return the dtype for each of the columns for the dataset"""
        raise NotImplementedError

    def load_to_memory(self):
        """
        Load data from zarr files to memeory

        Warning: Make sure the uncompressed dataset fits in-memory.
            This method will load the **uncompressed** dataset into memory. Make
            sure you actually have enough memory to store the dataset.
        """
        data = self.zarr_data

        if not isinstance(data, zarr.Group):
            raise TypeError(
                "The dataset zarr_root is not a valid Zarr archive. "
                "Did you call Dataset.load_to_memory() twice?"
            )

        # NOTE (cwognum): If the dataset fits in memory, the most performant is to use plain NumPy arrays.
        # Even if we disable chunking and compression in Zarr.
        # For more information, see https://github.com/zarr-developers/zarr-python/issues/1395
        self._zarr_data = load_zarr_group_to_memory(data)

    @abc.abstractmethod
    def get_data(self, row: str | int, col: str, adapters: Optional[List[Adapter]] = None) -> np.ndarray:
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

    @abc.abstractmethod
    def upload_to_hub(
        self, access: Optional[AccessType] = "private", owner: Union[HubOwner, str, None] = None
    ):
        """Uploads the dataset to the Polaris Hub."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, path: str):
        """Loads a benchmark from a JSON file.
        Overrides the method from the base class to remove the caching dir from the file to load from,
        as that should be user dependent.

        Args:
            path: Loads a benchmark specification from a JSON file.
        """
        with fsspec.open(path, "r") as f:
            data = json.load(f)
        data.pop("cache_dir", None)
        return cls.model_validate(data)

    @abc.abstractmethod
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

    def cache(self, cache_dir: Optional[str] = None, verify_checksum: bool = True) -> str:
        """Caches the dataset by downloading all additional data for pointer columns to a local directory.

        Args:
            cache_dir: The directory to cache the data to. If not provided,
                this will fall back to the `Dataset.cache_dir` attribute
            verify_checksum: Whether to verify the checksum of the dataset after caching.

        Returns:
            The path to the cache directory.
        """

        if cache_dir is not None:
            self.cache_dir = cache_dir

        self.to_json(self.cache_dir, load_zarr_from_new_location=True)

        if verify_checksum:
            self.verify_checksum()

        return self.cache_dir

    def size(self):
        return self.rows, self.n_columns

    def __getitem__(self, item):
        """Allows for indexing the dataset directly"""
        ret = self.table.loc[item]
        if isinstance(ret, pd.Series):
            # Load the data from the pointer columns

            if ret.name in self.table.columns:
                # Returning a column, the indices are rows
                if self.annotations[ret.name].is_pointer:
                    ret = np.array([self.get_data(k, ret.name) for k in ret.index])

            elif len(ret) == self.n_rows:
                # Returning a row, the indices are columns
                ret = {
                    k: self.get_data(k, ret.name) if self.annotations[ret.name].is_pointer else ret[k]
                    for k in ret.index
                }

        # Returning a dataframe
        if isinstance(ret, pd.DataFrame):
            for c in ret.columns:
                if self.annotations[c].is_pointer:
                    ret[c] = [self.get_data(item, c) for item in ret.index]
            return ret

        return ret

    @abc.abstractmethod
    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        raise NotImplementedError

    def _repr_html_(self):
        """For pretty-printing in Jupyter Notebooks"""
        return dict2html(self._repr_dict_())

    def __len__(self):
        return self.n_rows

    def __repr__(self):
        return json.dumps(self._repr_dict_(), indent=2)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Whether two datasets are equal is solely determined by the checksum"""
        if not isinstance(other, BaseDataset):
            return False
        return self.md5sum == other.md5sum

    def __del__(self):
        """Close the connection of the client"""
        if self._client is not None:
            self._client.close()
