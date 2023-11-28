import json
import os.path
from collections import defaultdict
from hashlib import md5
from typing import Dict, Literal, Optional, Tuple, Union

import fsspec
import numpy as np
import pandas as pd
import zarr
from loguru import logger
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from polaris._artifact import BaseArtifactModel
from polaris.dataset._column import ColumnAnnotation
from polaris.hub.settings import PolarisHubSettings
from polaris.utils import fs
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidDatasetError, PolarisChecksumError
from polaris.utils.io import get_zarr_root, robust_copy
from polaris.utils.types import AccessType, HttpUrlString, HubOwner, License

# Constants
_SUPPORTED_TABLE_EXTENSIONS = ["parquet"]
_CACHE_SUBDIR = "datasets"
_INDEX_SEP = "#"
_INDEX_FMT = f"{{path}}{_INDEX_SEP}{{index}}"


class Dataset(BaseArtifactModel):
    """Basic data-model for a Polaris dataset, implemented as a [Pydantic](https://docs.pydantic.dev/latest/) model.

    At its core, a dataset in Polaris is a tabular data structure that stores data-points in a row-wise manner.
    A Dataset can have multiple modalities or targets, can be sparse and can be part of one or multiple
     [`BenchmarkSpecification`][polaris.benchmark.BenchmarkSpecification] objects.

    Info: Pointer columns
        Whereas a `Dataset` contains all information required to construct a dataset, it is not ready yet.
        For complex data, such as images, we support storing the content in external blobs of data.
        In that case, the table contains _pointers_ to these blobs that are dynamically loaded when needed.

    Attributes:
        table: The core data-structure, storing data-points in a row-wise manner. Can be specified as either a
            path to a `.parquet` file or a `pandas.DataFrame`.
        md5sum: The checksum is used to verify the version of the dataset specification. If specified, it will
            raise an error if the specified checksum doesn't match the computed checksum.
        readme: Markdown text that can be used to provide a formatted description of the dataset.
            If using the Polaris Hub, it is worth noting that this field is more easily edited through the Hub UI
            as it provides a rich text editor for writing markdown.
        annotations: Each column _can be_ annotated with a [`ColumnAnnotation`][polaris.dataset.ColumnAnnotation] object.
            Importantly, this is used to annotate whether a column is a pointer column.
        source: The data source, e.g. a DOI, Github repo or URI.
    For additional meta-data attributes, see the [`BaseArtifactModel`][polaris._artifact.BaseArtifactModel] class.

    Raises:
        InvalidDatasetError: If the dataset does not conform to the Pydantic data-model specification.
        PolarisChecksumError: If the specified checksum does not match the computed checksum.
    """

    # Public attributes
    # Data
    table: Union[pd.DataFrame, str]
    md5sum: Optional[str] = None

    # Additional meta-data
    readme: str = ""
    annotations: Dict[str, ColumnAnnotation] = Field(default_factory=dict)
    source: Optional[HttpUrlString] = None
    license: Optional[License] = None

    # Config
    cache_dir: Optional[str] = None  # Where to cache the data to if cache() is called.

    # Private attributes
    _path_to_hash: Dict[str, Dict[str, str]] = defaultdict(dict)
    _has_been_warned: bool = False
    _has_been_cached: bool = False

    @field_validator("table")
    def _validate_table(cls, v):
        """If the table is not a dataframe yet, assume it's a path and try load it."""
        if not isinstance(v, pd.DataFrame):
            if not fs.is_file(v) or fs.get_extension(v) not in _SUPPORTED_TABLE_EXTENSIONS:
                raise InvalidDatasetError(f"{v} is not a valid DataFrame or .parquet path.")
            v = pd.read_parquet(v)
        return v

    @model_validator(mode="after")
    @classmethod
    def _validate_model(cls, m: "Dataset"):
        """If a checksum is provided, verify it matches what the checksum should be.
        If no checksum is provided, make sure it is set.
        If no cache_dir is provided, set it to the default cache dir and make sure it exists
        """

        # Verify that all annotations are for columns that exist
        if any(k not in m.table.columns for k in m.annotations):
            raise InvalidDatasetError("There is annotations for columns that do not exist")

        # Set a default for missing annotations and convert strings to Modality
        for c in m.table.columns:
            if c not in m.annotations:
                m.annotations[c] = ColumnAnnotation()

        # Verify the checksum
        # NOTE (cwognum): Is it still reasonable to always verify this as the dataset size grows?
        actual = m.md5sum
        expected = cls._compute_checksum(m.table)

        if actual is None:
            m.md5sum = expected
        elif actual != expected:
            raise PolarisChecksumError(
                "The dataset md5sum does not match what was specified in the meta-data. "
                f"{actual} != {expected}"
            )

        # Set the default cache dir if none and make sure it exists
        if m.cache_dir is None:
            m.cache_dir = fs.join(DEFAULT_CACHE_DIR, _CACHE_SUBDIR, m.name, m.md5sum)
        fs.mkdir(m.cache_dir, exist_ok=True)

        return m

    @staticmethod
    def _compute_checksum(table):
        """Computes a hash of the dataset.

        This is meant to uniquely identify the dataset and can be used to verify the version.

        1. Is not sensitive to the ordering of the columns or rows in the table.
        2. Purposefully does not include the meta-data (source, description, name, annotations).
        3. For any pointer column, it uses a hash of the path instead of the file contents.
            This is a limitation, but probably a reasonable assumption that helps practicality.
            A big downside is that as the dataset is saved elsewhere, the hash changes.
        """
        hash_fn = md5()

        # Sort the columns s.t. the checksum is not sensitive to the column-ordering
        df = table.copy(deep=True)
        df = df[sorted(df.columns.tolist())]

        # Use the sum of the row-wise hashes s.t. the hash is insensitive to the row-ordering
        table_hash = pd.util.hash_pandas_object(df).sum()
        hash_fn.update(table_hash)

        checksum = hash_fn.hexdigest()
        return checksum

    def get_data(self, row: Union[str, int], col: str) -> np.ndarray:
        """Since the dataset might contain pointers to external files, data retrieval is more complicated
        than just indexing the `table` attribute. This method provides an end-point for seamlessly
        accessing the underlying data.

        Args:
            row: The row index in the `Dataset.table` attribute
            col: The column index in the `Dataset.table` attribute

        Returns:
            A numpy array with the data at the specified indices. If the column is a pointer column,
                the content of the referenced file is loaded to memory.
        """

        def _load(p: str, index: Optional[int]) -> np.ndarray:
            """Tiny helper function to reduce code repetition."""
            arr = zarr.convenience.load(p)
            if index is not None:
                arr = arr[index]
            return arr

        value = self.table.loc[row, col]
        if not self.annotations[col].is_pointer:
            return value

        value, index = self._split_index_from_path(value)

        # In the case it is a pointer column, we need to load additional data into memory
        # We first check if the data has been downloaded to the cache.
        path = self._get_cache_path(column=col, value=value)
        if fs.exists(path):
            return _load(path, index)

        # If it doesn't exist, we load from the original path and warn if not local
        if not fs.is_local_path(value) and not self._has_been_warned:
            logger.warning(
                f"You're loading data from a remote location. "
                f"To speed up this process, consider caching the dataset first "
                f"using {self.__class__.__name__}.cache()"
            )
            self._has_been_warned = True
        return _load(value, index)

    def upload_to_hub(
        self,
        env_file: Optional[Union[str, os.PathLike]] = None,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        access: Optional[AccessType] = "private",
        owner: Optional[Union[HubOwner, str]] = None,
        **kwargs: dict,
    ):
        """
        Very light, convenient wrapper around the
        [`PolarisHubClient.upload_dataset`][polaris.hub.client.PolarisHubClient.upload_dataset] method.
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            env_file=env_file, settings=settings, cache_auth_token=cache_auth_token, **kwargs
        ) as client:
            return client.upload_dataset(self, access=access, owner=owner)

    @classmethod
    def from_zarr(cls, path: str) -> "Dataset":
        """Parse a [.zarr](https://zarr.readthedocs.io/en/stable/index.html) hierarchy into a Polaris `Dataset`.

        In short: A `.zarr` file can contain groups and arrays, where each group can again contain groups and arrays.
        Additional user attributes (for any array or group) are saved as JSON files.

        Within Polaris:

        1. Each subgroup of the root group corresponds to a single column.
        2. Each subgroup can contain:
            - A single array with _all_ datapoints.
            - A single array _per_ datapoint.
        3. Additional meta-data is saved to the user attributes of the root group.
        3. The indices are required to be integers.

        Tip: Tutorial
            To learn more about the zarr format, see the
            [tutorial](../tutorials/dataset_zarr.ipynb).

        Warning: Beta functionality
            This feature is still in beta and the API will likely change. Please report any issues you encounter.

        Args:
            path: The path to the root of the `.zarr` directory. Should be compatible with fsspec.
        """

        logger.warning(
            "We are still testing to save and load from .zarr files. "
            "This part of the API will likely change."
        )

        root = zarr.open(path, "r")

        # Get the user attributes
        attrs = root.attrs.asdict()

        # TODO (cwognum): This is outdated and needs to be updated.
        possible_user_attr = ["name", "description", "source", "annotations"]
        attrs = {k: v for k, v in attrs.items() if k in possible_user_attr}

        # Set the annotations
        attrs["annotations"] = attrs.get("annotations", {})
        for column_label in root.group_keys():
            obj = attrs["annotations"].get(column_label, {})
            obj = ColumnAnnotation.model_validate(obj)
            obj.is_pointer = True
            attrs["annotations"][column_label] = obj

        # Construct the table
        # Parse any group into a column
        data = defaultdict(dict)
        for col, group in root.groups():
            keys = list(group.array_keys())

            if len(keys) == 1:
                arr = group[keys[0]]
                for i, arr_row in enumerate(arr):
                    # In case all data is saved in a single array, we construct a path with an index suffix.
                    data[col][i] = _INDEX_FMT.format(path=fs.join(path, arr.name), index=i)

            else:
                for name, arr in group.arrays():
                    try:
                        name = int(name)
                    except ValueError as error:
                        raise InvalidDatasetError(
                            "All names for arrays in the .zarr archive are required to be integers."
                        ) from error
                    data[col][name] = fs.join(path, arr.path)

        # Construct the dataset
        table = pd.DataFrame(data)
        return cls(table=table, **attrs)

    def to_zarr(
        self,
        destination: str,
        array_mode: Dict[str, Literal["single", "multiple"]],
    ) -> str:
        """Saves a dataset to a .zarr file. For more information on the resulting structure,
        see [`from_zarr`][polaris.dataset.Dataset.from_zarr].

        Tip: Tutorial
            To learn more about the zarr format, see the
            [tutorial](../tutorials/dataset_zarr.ipynb).

        Warning: Beta functionality
            This feature is still in beta and the API will likely change. Please report any issues you encounter.

        Args:
            destination: The _directory_ to save the associated data to.
            array_mode: For each of the columns, whether to save all datapoints in a single array
                or create an array per datapoint. Should be one of "single" or "multiple".

        Returns:
            The path to the root zarr file.
        """

        logger.warning(
            "We are still testing to save and load from .zarr files. "
            "This part of the API will likely change."
        )

        if array_mode not in ["single", "multiple"]:
            raise ValueError(f"array_mode should be one of 'single' or 'multiple', not {array_mode}")

        fs.mkdir(destination, exist_ok=True)
        path = fs.join(destination, "dataset.zarr")

        if not isinstance(array_mode, dict):
            array_mode = {k: array_mode for k in self.table.columns}

        root = zarr.open(path, "w")
        for col in self.table.columns:
            group = root.create_group(col)

            # Load an example to get the dtype and shape
            example = self.get_data(row=0, col=col)

            if array_mode[col] == "single":
                # Create one big array for all datapoints
                shape = (len(self.table), *example.shape)
                arr = group.empty(col, shape=shape, dtype=example.dtype)

                for row in self.table.index:
                    # Save the data to the array
                    arr[row] = self.get_data(row=row, col=col)
            else:
                for row in self.table.index:
                    # Create an array per datapoint
                    group.array(row, self.get_data(row=row, col=col))

        # Save the meta-data
        # TODO (cwognum): This is outdated and needs to be updated.
        root.user_attrs = {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "annotations": {k: v.model_dump() for k, v in self.annotations.items()},
        }
        return path

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

    def to_json(self, destination: str) -> str:
        """
        Save the dataset to a destination directory as a JSON file.

        Warning: Multiple files
            Perhaps unintuitive, this method creates multiple files.

            1. `/path/to/destination/dataset.json`: This file can be loaded with
                [`Dataset.from_json`][polaris.dataset.Dataset.from_json].
            2. `/path/to/destination/table.parquet`: The `Dataset.table` attribute is saved here.
            3. _(Optional)_ `/path/to/destination/data/*`: Any additional blobs of data referenced by the
                    pointer columns will be stored here.

        Args:
            destination: The _directory_ to save the associated data to.

        Returns:
            The path to the JSON file.
        """
        fs.mkdir(destination, exist_ok=True)
        table_path = fs.join(destination, "table.parquet")
        dataset_path = fs.join(destination, "dataset.json")
        pointer_dir = fs.join(destination, "data")

        # Save additional data
        new_table = self._copy_and_update_pointers(pointer_dir, inplace=False)

        # Lu: Avoid serilizing and sending None to hub app.
        serialized = self.model_dump(exclude={"cache_dir"}, exclude_none=True)
        serialized["table"] = table_path

        # We need to recompute the checksum, as the pointer paths have changed
        serialized["md5sum"] = self._compute_checksum(new_table)

        new_table.to_parquet(table_path)
        with fsspec.open(dataset_path, "w") as f:
            json.dump(serialized, f)

        return dataset_path

    def cache(self, cache_dir: Optional[str] = None) -> str:
        """Caches the dataset by downloading all additional data for pointer columns to a local directory.

        Args:
            cache_dir: The directory to cache the data to. If not provided,
                this will fall back to the `Dataset.cache_dir` attribute

        Returns:
            The path to the cache directory.
        """

        if cache_dir is not None:
            self.cache_dir = cache_dir

        if not self._has_been_cached:
            self._copy_and_update_pointers(self.cache_dir, inplace=True)
            self._has_been_cached = True
        return self.cache_dir

    def _get_cache_path(self, column: str, value: str) -> Optional[str]:
        """
        Returns where the data _would be_ cached for any entry in the pointer columns,
        or None if the column is not a pointer column.
        """
        if not self.annotations[column].is_pointer:
            return

        if value not in self._path_to_hash[column]:
            h = md5(value.encode("utf-8")).hexdigest()

            value, _ = self._split_index_from_path(value)
            ext = fs.get_extension(value)
            dst = fs.join(self.cache_dir, column, f"{h}.{ext}")

            # The reason for caching the path is to speed-up retrieval. Hashing can be slow and with large
            # datasets this could become a bottleneck.
            self._path_to_hash[column][value] = dst

        return self._path_to_hash[column][value]

    def size(self):
        return len(self), len(self.table.columns)

    def _split_index_from_path(self, path: str) -> Tuple[str, Optional[int]]:
        """
        Paths can have an additional index appended to them.
        This extracts that index from the path.
        """
        index = None
        if _INDEX_SEP in path:
            path, index = path.split(_INDEX_SEP)
            index = int(index)
        return path, index

    def _copy_and_update_pointers(
        self, save_dir: str, table: Optional[pd.DataFrame] = None, inplace: bool = False
    ) -> pd.DataFrame:
        """Copy and update the path in the table to the new destination"""

        def fn(path):
            """Helper function that can be used within Pandas apply to copy and update all files"""

            # We copy the entire .zarr hierarchy
            root = get_zarr_root(path)
            if root is None:
                raise NotImplementedError(
                    "Only the .zarr file format is currently supported for pointer columns"
                )

            # We could introduce name collisions here and thus use a hash of the original path for the destination
            dst = fs.join(save_dir, f"{md5(root.encode('utf-8')).hexdigest()}.zarr")
            robust_copy(root, dst)

            diff = os.path.relpath(path, root)
            dst = fs.join(dst, diff)
            return dst

        if table is None:
            table = self.table
        if not inplace:
            table = self.table.copy(deep=True)

        for c in table.columns:
            if self.annotations[c].is_pointer:
                table[c] = table[c].apply(fn)
        return table

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        repr_dict = self.model_dump()
        repr_dict.pop("table")
        return repr_dict

    def _repr_html_(self):
        """For pretty-printing in Jupyter Notebooks"""
        return dict2html(self._repr_dict_())

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return json.dumps(self._repr_dict_(), indent=2)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Whether two datasets are equal is solely determined by the checksum"""
        if not isinstance(other, Dataset):
            return False
        return self.md5sum == other.md5sum
