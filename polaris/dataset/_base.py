import abc
import json
from os import PathLike
from pathlib import Path, PurePath
from typing import Any, Iterable, MutableMapping
from uuid import uuid4

import fsspec
import numpy as np
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
from typing_extensions import Self

from polaris._artifact import BaseArtifactModel
from polaris.dataset._adapters import Adapter
from polaris.dataset._column import ColumnAnnotation
from polaris.dataset.zarr import MemoryMappedDirectoryStore
from polaris.dataset.zarr._utils import check_zarr_codecs, load_zarr_group_to_memory
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidDatasetError
from polaris.utils.types import (
    AccessType,
    ChecksumStrategy,
    DatasetIndex,
    HttpUrlString,
    HubOwner,
    SupportedLicenseType,
    ZarrConflictResolution,
)

# Constants
_CACHE_SUBDIR = "datasets"


class BaseDataset(BaseArtifactModel, abc.ABC):
    """Base data-model for a Polaris dataset, implemented as a [Pydantic](https://docs.pydantic.dev/latest/) model.

    At its core, a dataset in Polaris can _conceptually_ be thought of as tabular data structure that stores data-points
    in a row-wise manner, where each column correspond to a variable associated with that datapoint.

    A Dataset can have multiple modalities or targets, can be sparse and can be part of one or multiple
     [`BenchmarkSpecification`][polaris.benchmark.BenchmarkSpecification] objects.

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

    For additional meta-data attributes, see the base classes.

    Raises:
        InvalidDatasetError: If the dataset does not conform to the Pydantic data-model specification.
    """

    # Public attributes
    # Data
    default_adapters: dict[str, Adapter] = Field(default_factory=dict)
    zarr_root_path: str | None = None

    # Additional meta-data
    readme: str = ""
    annotations: dict[str, ColumnAnnotation] = Field(default_factory=dict)
    source: HttpUrlString | None = None
    license: SupportedLicenseType | None = None
    curation_reference: HttpUrlString | None = None

    # Private attributes
    _zarr_root: zarr.Group | None = PrivateAttr(None)
    _zarr_data: MutableMapping[str, np.ndarray] | None = PrivateAttr(None)
    _warn_about_remote_zarr: bool = PrivateAttr(True)
    _cache_dir: str | None = PrivateAttr(None)  # Where to cache the data to if cache() is called.
    _verify_checksum_strategy: ChecksumStrategy = PrivateAttr("verify_unless_zarr")

    @field_validator("default_adapters", mode="before")
    def _validate_adapters(cls, value):
        """Validate the adapters"""
        return {k: Adapter[v] if isinstance(v, str) else v for k, v in value.items()}

    @field_serializer("default_adapters")
    def _serialize_adapters(self, value: dict[str, Adapter]) -> dict[str, str]:
        """Serializes the adapters"""
        return {k: v.name for k, v in value.items()}

    @model_validator(mode="after")
    def _validate_base_dataset_model(self) -> Self:
        # Verify that all annotations are for columns that exist
        if any(k not in self.columns for k in self.annotations):
            raise InvalidDatasetError(
                f"There are annotations for columns that do not exist. Columns: {self.columns}. Annotations: {list(self.annotations.keys())}"
            )

        # Verify that all adapters are for columns that exist
        if any(k not in self.columns for k in self.default_adapters.keys()):
            raise InvalidDatasetError(
                f"There are default adapters for columns that do not exist. Columns: {self.columns}. Adapters: {list(self.annotations.keys())}"
            )

        # Set a default for missing annotations and convert strings to Modality
        for c in self.columns:
            if c not in self.annotations:
                self.annotations[c] = ColumnAnnotation()
            self.annotations[c].dtype = self.dtypes[c]

        return self

    @model_validator(mode="after")
    def _ensure_cache_dir_exists(self) -> Self:
        """
        Set the default cache dir if none and make sure it exists
        """
        if self._cache_dir is None:
            self._cache_dir = str(PurePath(DEFAULT_CACHE_DIR) / _CACHE_SUBDIR / str(uuid4()))
        fs, path = fsspec.url_to_fs(self._cache_dir)
        fs.mkdirs(path, exist_ok=True)

        return self

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
    def zarr_root(self) -> zarr.Group | None:
        """Get the zarr Group object corresponding to the root.

        Opens the zarr archive in read-write mode if it is not already open.

        Note: Different to `zarr_data`
            The `zarr_data` attribute references either to the Zarr archive or to a in-memory copy of the data.
            See also [`Dataset.load_to_memory`][polaris.dataset.Dataset.load_to_memory].
        """

        from polaris.hub.storage import StorageSession

        if self._zarr_root is not None:
            return self._zarr_root

        if self.zarr_root_path is None:
            return None

        saved_on_hub = self.zarr_root_path.startswith(StorageSession.polaris_protocol)

        if self._warn_about_remote_zarr and saved_on_hub:
            # TODO (cwognum): The user now has no easy way of knowing whether the dataset is "small enough".
            logger.warning(
                f"You're loading data from a remote location. "
                f"If the dataset is small enough, consider caching the dataset first "
                f"using {self.__class__.__name__}.cache() for more performant data access."
            )
            self._warn_about_remote_zarr = False

        try:
            if saved_on_hub:
                self._zarr_root = self.load_zarr_root_from_hub()
            else:
                self._zarr_root = self.load_zarr_root_from_local()

        except KeyError as error:
            raise InvalidDatasetError(
                "A Zarr archive associated with a Polaris dataset has to be consolidated."
            ) from error

        check_zarr_codecs(self._zarr_root)
        return self._zarr_root

    @abc.abstractmethod
    def load_zarr_root_from_hub(self):
        """
        Loads a Zarr archive from the Polaris Hub.
        """
        raise NotImplementedError

    def load_zarr_root_from_local(self):
        """
        Loads a locally stored Zarr archive.

        We use memory mapping by default because our experiments show that it's consistently faster
        """
        if not Path(self.zarr_root_path).exists():
            raise FileNotFoundError(
                f"The Zarr archive at {self.zarr_root_path} does not exist. "
                "If the data is not stored on the Polaris Hub, it needs to be stored locally."
            )
        store = MemoryMappedDirectoryStore(self.zarr_root_path)
        return zarr.open_consolidated(store, mode="r+")

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
    def rows(self) -> Iterable[str | int]:
        """Return all row indices for the dataset"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def columns(self) -> list[str]:
        """Return all columns for the dataset"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtypes(self) -> dict[str, np.dtype]:
        """Return the dtype for each of the columns for the dataset"""
        raise NotImplementedError

    @abc.abstractmethod
    def should_verify_checksum(self, strategy: ChecksumStrategy) -> bool:
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
        raise NotImplementedError

    @abc.abstractmethod
    def upload_to_hub(self, access: AccessType = "private", owner: HubOwner | str | None = None):
        """Uploads the dataset to the Polaris Hub."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_json(cls, path: str):
        """
        Loads a dataset from a JSON file.

        Args:
            path: The path to the JSON file to load the dataset from.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_json(self, destination: str | Path, if_exists: ZarrConflictResolution = "replace") -> str:
        """
        Save the dataset to a destination directory as a JSON file.

        Args:
            destination: The _directory_ to save the associated data to.
            if_exists: Action for handling existing files in the Zarr archive. Options are 'raise' to throw
                an error, 'replace' to overwrite, or 'skip' to proceed without altering the existing files.

        Returns:
            The path to the JSON file.
        """
        raise NotImplementedError

    def size(self) -> tuple[int, int]:
        return self.n_rows, self.n_columns

    def __getitem__(self, item: DatasetIndex) -> Any | np.ndarray | dict[str, np.ndarray]:
        """Allows for indexing the dataset directly"""

        # If a tuple, we assume it's the row and column index pair
        if isinstance(item, tuple):
            row, col = item
            return self.get_data(row, col)

        # Otherwise, we assume you're indexing the row
        return {col: self.get_data(item, col) for col in self.columns}

    @abc.abstractmethod
    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        raise NotImplementedError

    def _repr_html_(self) -> str:
        """For pretty-printing in Jupyter Notebooks"""
        return dict2html(self._repr_dict_())

    def __len__(self):
        return self.n_rows

    def __repr__(self):
        return json.dumps(self._repr_dict_(), indent=2)

    def __str__(self):
        return self.__repr__()

    def _cache_zarr(self, destination: str | PathLike, if_exists: ZarrConflictResolution):
        """
        Caches the Zarr archive to a local directory.
        """
        from polaris.hub.storage import S3Store

        if not self.uses_zarr:
            return

        current_zarr_root = PurePath(self.zarr_root_path)
        destination_zarr_root = PurePath(destination) / current_zarr_root.name

        # Copy over Zarr data to the destination
        self._warn_about_remote_zarr = False

        logger.info(f"Copying Zarr archive to {destination_zarr_root}. This may take a while.")
        destination_store = zarr.open(str(destination_zarr_root), "w").store
        source_store = self.zarr_root.store.store

        if isinstance(source_store, S3Store):
            source_store.copy_to_destination(destination_store, if_exists, logger.info)
        else:
            zarr.copy_store(
                source=source_store,
                dest=destination_store,
                log=logger.info,
                if_exists=if_exists,
            )
        self.zarr_root_path = str(destination_zarr_root)
        self._zarr_root = None
        self._zarr_data = None
