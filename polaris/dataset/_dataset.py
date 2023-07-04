import os.path
import string

import yaml
import zarr
import zarr.convenience
import fsspec
import json

import numpy as np
import pandas as pd

from hashlib import md5
from collections import defaultdict
from typing import Dict, Optional, Union, Tuple
from pydantic import BaseModel, PrivateAttr, field_validator, model_validator, computed_field
from loguru import logger

from polaris.utils import fs
from polaris.dataset._column import ColumnAnnotation, Modality
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.io import robust_copy, get_zarr_root
from polaris.utils.errors import InvalidDatasetError, PolarisChecksumError
from polaris.utils.dict2html import dict2html


class Dataset(BaseModel):
    """
    A dataset class is a wrapper around a pandas DataFrame.

    It is a tabular format which contains all information to construct an ML-dataset, but is not ready yet.
    For example, if the dataset contains the Image modality, the image data is not yet loaded but the column
    rather contains a pointer to the image file.

    A Dataset can have multiple modalities or targets, can be sparse and can be part of one or multiple Benchmarks.
    """

    _SUPPORTED_TABLE_EXTENSIONS = ["parquet"]
    _CACHE_SUBDIR = "datasets"
    _INDEX_SEP = "#"
    _INDEX_FMT = f"{{path}}{_INDEX_SEP}{{index}}"

    """
    The table stores row-wise datapoints
    """
    table: Union[pd.DataFrame, str]

    """
    The public-facing name of the dataset
    """
    name: str

    """
    A beginner-friendly description of the dataset
    """
    description: str

    """
    The data source, e.g. a DOI, Github repo or URI
    """
    source: str

    """
    Annotates each column with the modality of that column and additional meta-data
    """
    annotations: Dict[str, Union[str, Modality, ColumnAnnotation]] = {}

    """
    Hash of the dataset, used to verify that the dataset is the expected version
    """
    md5sum: Optional[str] = None

    """
    Where the dataset is cached to locally
    """
    cache_dir: Optional[str] = None

    """
    The dataset URL on the Polaris Hub.
    """

    @computed_field
    @property
    def polaris_hub_url(self) -> Optional[str]:
        # NOTE(hadim): putting as default here but we could make it optional
        return "https://polaris.io/dataset/ORG_OR_USER/DATASET_NAME?"

    _path_to_hash: Dict[str, Dict[str, str]] = PrivateAttr(defaultdict(dict))
    _has_been_warned: bool = PrivateAttr(False)
    _has_been_cached: bool = PrivateAttr(False)

    class Config:
        arbitrary_types_allowed = True

    @field_validator("table")
    def validate_table(cls, v):
        """If the table is not a dataframe yet, assume it's a path and try load it."""
        if not isinstance(v, pd.DataFrame):
            if not fs.is_file(v) or fs.get_extension(v) not in cls._SUPPORTED_TABLE_EXTENSIONS:
                raise InvalidDatasetError(f"{v} is not a valid DataFrame or .parquet path.")
            v = pd.read_parquet(v)
        return v

    @field_validator("name")
    def validate_name(cls, v):
        """
        Verify the name only contains valid characters which can be used in a file path.
        """
        valid_characters = string.ascii_letters + string.digits + "_-"
        if not all(c in valid_characters for c in v):
            raise InvalidDatasetError(f"`name` can only contain alpha-numeric characters, - or _, found {v}")
        return v

    @field_validator("annotations")
    def validate_annotations(cls, v, values):
        """
        Set missing annotations to default.
        For all columns that contain pointers, verify all paths listed exist.
        """
        # Exit early if a previous validation step failed
        if "table" not in values:
            return v

        # Verify that all annotations are for columns that exist
        if any(k not in values["table"].columns for k in v):
            raise InvalidDatasetError("There is annotations for columns that do not exist")

        # Set a default for missing annotations and convert strings to Modality
        for c in values["table"].columns:
            if c not in v:
                v[c] = ColumnAnnotation()
            if isinstance(v[c], (str, Modality)):
                v[c] = ColumnAnnotation(modality=v[c])
        return v

    # TODO(hadim): see https://github.com/pydantic/pydantic/issues/6277
    # and https://github.com/pydantic/pydantic/pull/6279
    @model_validator(mode="after")
    @classmethod
    def validate_checksum_and_cache_dir(cls, values):
        """
        If a checksum is provided, verify it matches what the checksum should be.
        If no checksum is provided, make sure it is set.
        If no cache_dir is provided, set it to the default cache dir and make sure it exists
        """

        # This is still ran when some fields are not specified
        if any(k not in values for k in ["table", "annotations", "name"]) or len(values["annotations"]) == 0:
            return

        # Verify the checksum
        # NOTE (cwognum): Is it still reasonable to always verify this as the dataset size grows?
        actual = values["md5sum"]
        expected = cls._compute_checksum(values["table"], values["annotations"])

        if actual is None:
            values["md5sum"] = expected
        elif actual != expected:
            raise PolarisChecksumError(
                "The dataset md5sum does not match what was specified in the meta-data. "
                f"{actual} != {expected}"
            )

        # Set the default cache dir if none and make sure it exists
        if values["cache_dir"] is None:
            values["cache_dir"] = fs.join(
                DEFAULT_CACHE_DIR, cls._CACHE_SUBDIR, values["name"], values["md5sum"]
            )
        fs.mkdir(values["cache_dir"], exist_ok=True)

        return values

    @classmethod
    def from_yaml(cls, path: str) -> "Dataset":
        """
        Loads a dataset from a yaml file.
        This is a very thin wrapper around pydantic's model_validate.
        """
        with fsspec.open(path, "r") as fd:
            data = yaml.safe_load(fd)
        return Dataset.model_validate(data)

    @classmethod
    def from_zarr(cls, path: str):
        """
        Parse a zarr hierarchy into a Dataset.

        Each group in the zarr hierarchy is parsed into a pointer column.
        Meta-data and additional columns can be specified in the user attributes.
        To specify additional columns, we use a dict with the column -> index -> value format.
        """

        expected_user_attrs = ["name", "description", "source", "annotations"]

        root = zarr.open_group(path, "r")
        if any(k not in root.attrs for k in expected_user_attrs):
            raise InvalidDatasetError(
                f"To load a dataset from a .zarr hierarchy, the root group must contain "
                f"the following attributes: {expected_user_attrs}. Found {list(root.attrs.keys())}"
            )

        # Construct the table
        # Parse any group into a pointer column
        data = defaultdict(dict)
        for col, group in root.groups():
            keys = list(group.array_keys())

            if len(keys) == 1:
                arr = group[keys[0]]
                for i, arr_row in enumerate(arr):
                    data[col][i] = cls._INDEX_FMT.format(path=fs.join(path, arr.name), index=i)

            else:
                for name, arr in group.arrays():
                    try:
                        name = int(name)
                    except ValueError as error:
                        raise TypeError(f"Array names in .zarr hierarchies should be integers") from error
                    data[col][name] = fs.join(path, arr.path)

        # Parse additional columns from the user attributes
        attrs = root.attrs.asdict()
        cols = list(attrs.keys())
        for col in cols:
            if col not in expected_user_attrs:
                if not isinstance(attrs[col], dict):
                    raise TypeError(
                        f"Expected the dictionary type for user attr `{col}`, found {type(attrs[col])}"
                    )
                # All non-expected user attrs are assumed to be additional columns
                # These should be specified as dicts with index -> value
                d = {}
                for k, v in attrs.pop(col).items():
                    try:
                        k = int(k)
                    except ValueError as error:
                        raise TypeError(f"Column names in .zarr hierarchies should be integers") from error
                    d[k] = v
                data[col] = d

        # Construct the dataset
        table = pd.DataFrame(data)
        return cls(table=table, **attrs)

    @staticmethod
    def _compute_checksum(table, annotations):
        """
        Computes a hash of the dataset.

        This is meant to uniquely identify the dataset and can be used to verify the version.

        (1) Is not sensitive to the ordering of the columns or rows in the table.
        (2) Purposefully does not include the meta-data (source, description, name).
        (3) For any pointer column, it uses a hash of the path instead of the file contents.
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

        for c in df.columns:
            modality = annotations[c].modality.name.encode("utf-8")
            hash_fn.update(modality)

        checksum = hash_fn.hexdigest()
        return checksum

    def __len__(self):
        """Return the number of datapoints"""
        return len(self.table)

    def _repr_dict_(self) -> dict:
        repr_dict = self.model_dump()
        repr_dict.pop("table")

        repr_dict["annotations"] = {}
        for k, v in self.annotations.items():
            repr_dict["annotations"][k] = v.modality.name

        # TODO(hadim): remove once @compute_field is available
        repr_dict["polaris_hub_url"] = self.polaris_hub_url

        return repr_dict

    def __repr__(self):
        return json.dumps(self._repr_dict_(), indent=2)

    def _repr_html_(self):
        return dict2html(self._repr_dict_())

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Whether two datasets are equal is solely determined by the checksum"""
        if not isinstance(other, Dataset):
            return False
        return self.md5sum == other.md5sum

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
            if self.annotations[c].is_pointer():
                table[c] = table[c].apply(fn)
        return table

    def to_yaml(self, destination: str):
        """
        Save the dataset to a destination directory
        """
        fs.mkdir(destination, exist_ok=True)
        table_path = fs.join(destination, "table.parquet")
        dataset_path = fs.join(destination, "dataset.yaml")
        pointer_dir = fs.join(destination, "data")

        # Save additional data
        new_table = self._copy_and_update_pointers(pointer_dir, inplace=False)

        serialized = self.model_dump()
        serialized["table"] = table_path
        serialized["annotations"] = {k: m.model_dump() for k, m in self.annotations.items()}
        # We need to recompute the checksum, as the pointer paths have changed
        serialized["md5sum"] = self._compute_checksum(new_table, self.annotations)

        new_table.to_parquet(table_path)
        with fsspec.open(dataset_path, "w") as f:
            yaml.dump(serialized, f)

        return dataset_path

    def get_cache_path(self, column: str, value: str) -> str:
        """
        Returns where the data _would be_ cached for any entry in the pointer columns
        """
        if value not in self._path_to_hash[column]:
            h = md5(value.encode("utf-8")).hexdigest()

            value, _ = self._split_index_from_path(value)
            ext = fs.get_extension(value)
            dst = fs.join(self.cache_dir, column, f"{h}.{ext}")

            # The reason for caching the path is to speed-up retrieval. Hashing can be slow and with large
            # datasets this could become a bottleneck.
            self._path_to_hash[column][value] = dst

        return self._path_to_hash[column][value]

    def _split_index_from_path(self, path: str) -> Tuple[str, Optional[int]]:
        """
        Paths can have an additional index appended to them.
        This extracts that index from the path.
        """
        index = None
        if self._INDEX_SEP in path:
            path, index = path.split(self._INDEX_SEP)
            index = int(index)
        return path, index

    def get_data(self, row, col):
        """
        Helper function to get data from the Dataset. Since the dataset might contain pointers to
        external files, data retrieval is more complicated than just indexing the table.
        """

        def _load(p: str, index: Optional[int]) -> np.ndarray:
            # NOTE (cwognum): We will want to support more formats than just .zarr in the future
            #  which is why I have it in a separate function.
            arr = zarr.convenience.load(p)
            if index is not None:
                arr = arr[index]
            return arr

        value = self.table.loc[row, col]
        if not self.annotations[col].is_pointer():
            return value

        value, index = self._split_index_from_path(value)

        # In the case it is a pointer column, we need to load additional data into memory
        # We first check if the data has been downloaded to the cache.
        path = self.get_cache_path(column=col, value=value)
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

    def size(self):
        return len(self), len(self.table.columns)

    def cache(self) -> str:
        """
        Download all additional data for pointer columns.
        This does not change any data in the table itself, but populates a mapping from the table paths
        to the cached paths. This thus keeps the original dataset intact.
        """
        if not self._has_been_cached:
            self._copy_and_update_pointers(self.cache_dir, inplace=True)
            self._has_been_cached = True
        return self.cache_dir
