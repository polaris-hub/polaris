import yaml
import zarr
import fsspec

import numpy as np
import pandas as pd

from hashlib import md5
from collections import defaultdict
from typing import Dict, Optional, Union, List
from pydantic import BaseModel, Field, PrivateAttr, validator, root_validator
from loguru import logger

from polaris.utils import fs
from polaris.dataset import Modality
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.io import robust_file_copy
from polaris.utils.errors import InvalidDatasetError, PolarisChecksumError


class Dataset(BaseModel):
    """
    A dataset class is a wrapper around a pandas DataFrame.

    It is a tabular format which contains all information to construct an ML-dataset, but is not ready yet.
    For example, if the dataset contains the Image modality, the image data is not yet loaded but the column
    rather contains a pointer to the image file.

    A Dataset can have multiple modalities or targets, can be sparse and can be part of one or multiple Benchmarks.
    """

    _SUPPORTED_TABLE_EXTENSIONS = ["parquet"]

    # The table stores row-wise datapoints
    table: Union[pd.DataFrame, str]

    # The public-facing name of the dataset
    name: str

    # A beginner-friendly description of the dataset
    description: str

    # The data source, e.g. a DOI, Github repo or URI
    source: str

    # Annotates each column with the modality of that column
    modalities: Dict[str, Union[str, Modality]]

    # Hash of the dataset, used to verify that the dataset is the expected version
    checksum: Optional[str] = None

    # Where the dataset is cached to locally
    cache_dir: Optional[str] = Field(None, alias="cache_dir")

    _path_to_hash: Dict[str, Dict[str, str]] = PrivateAttr(defaultdict(dict))
    _has_been_warned: bool = PrivateAttr(False)

    class Config:
        arbitrary_types_allowed = True

    @validator("table")
    def validate_table(cls, v):
        if not isinstance(v, pd.DataFrame):
            if not fs.is_file(v) or fs.get_extension(v) not in cls._SUPPORTED_TABLE_EXTENSIONS:
                raise InvalidDatasetError(f" {v} is not a valid DataFrame or .parquet path.")
            v = pd.read_parquet(v)
        return v

    @validator("modalities")
    def validate_modalities(cls, v, values):
        # Fill missing modalities and convert strings to modalities
        for c in values["table"].columns:
            if c not in v:
                v[c] = Modality.UNKNOWN
            if not isinstance(v[c], Modality):
                v[c] = Modality[v[c].upper()]

            # Verify that pointer columns have valid extensions and exist
            if v[c].is_pointer():
                paths = values["table"][c].values
                if any(not fs.exists(p) for p in paths):
                    raise InvalidDatasetError(
                        f"The {c} [{v[c]}] column contains pointers "
                        f"that reference files which do not exist."
                    )
        return v

    @validator("cache_dir")
    def validate_cache_dir(cls, v, values):
        if v is None:
            v = fs.join(DEFAULT_CACHE_DIR, values["name"], values["checksum"])
        return v

    @root_validator
    def validate_checksum(cls, values):
        if not all(k in values for k in ["table", "modalities"]):
            # Skip validation as an earlier step has failed
            return values

        checksum = values["checksum"]
        table = values["table"]
        modalities = values["modalities"]

        expected = cls._compute_checksum(table, modalities)

        if checksum is None:
            values["checksum"] = expected
        elif checksum != expected:
            raise PolarisChecksumError(
                "The dataset checksum does not match what was specified in the meta-data. "
                f"{checksum} != {expected}"
            )
        return values

    @classmethod
    def from_yaml(cls, path: str) -> "Dataset":
        with fsspec.open(path, "r") as fd:
            data = yaml.safe_load(fd)
        return Dataset.parse_obj(data)

    @classmethod
    def from_zarr(cls, path: str):
        """
        Parse a zarr hierarchy into a Dataset.

        The expected zarr hierarchy should have a group per pointer modality
        and should save meta-data in the user attributes.
        """

        expected_user_attrs = ["name", "description", "source", "modalities"]

        root = zarr.open_group(path)
        if any(k not in root.attrs for k in expected_user_attrs):
            raise InvalidDatasetError(
                f"To load a dataset from a .zarr hierarchy, the root group must contain "
                f"the following attributes: {expected_user_attrs}. Found {list(root.attrs.keys())}"
            )

        # Construct the table
        data = defaultdict(dict)
        for col, group in root.groups():
            for name, arr in group.arrays():
                data[col][name] = fs.join(path, arr.path)
        for col in root.attrs:
            if col not in expected_user_attrs:
                if not isinstance(root.attrs[col], dict):
                    raise TypeError(
                        f"Expected the dictionary type for user attr `{col}`, found {type(root.attrs[col])}"
                    )
                # All non-expected user attrs are assumed to be additional columns
                # These should be specified as dictionaries: index -> value
                data[col] = root.attrs.pop(col)
        table = pd.DataFrame(data)

        return cls(table=table, **root.attrs)

    @staticmethod
    def _compute_checksum(table, modalities):
        # NOTE (cwognum): We do NOT hash the name, description and source.
        #  This is intentional, as I do not consider this meta-data to be important to the version.
        hash_fn = md5()
        # Reset the index and take the transpose to include column names
        hash_fn.update(pd.util.hash_pandas_object(table.reset_index().T, index=True).values)
        for c in table.columns:
            hash_fn.update(modalities[c].name.encode("utf-8"))
        checksum = hash_fn.hexdigest()
        return checksum

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        self.table.loc[""] = [f"[{self.modalities[c]}]" for c in self.table.columns]
        s = "\n"
        s += repr(self.table)
        s += f" [checksum {self.checksum}]"
        self.table.drop(self.table.tail(1).index, inplace=True)
        return s

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return False
        return self.checksum == other.checksum

    def save(self, destination: str):
        """
        Save the dataset to a destination directory
        """
        table_path = fs.join(destination, "table.parquet")
        dataset_path = fs.join(destination, "dataset.yaml")
        pointer_dir = fs.join(destination, "data")

        def _copy_and_update(src):
            ext = fs.get_extension(src)
            dst = fs.join(pointer_dir, f"{md5(src.encode('utf-8')).hexdigest()}.{ext}")
            robust_file_copy(src, dst)
            return dst

        # Save additional data
        new_table = self.table.copy(deep=True)

        for c in new_table.columns:
            if self.modalities[c].is_pointer():
                new_table[c] = new_table[c].apply(_copy_and_update)

        serialized = self.dict()
        serialized["table"] = table_path
        serialized["modalities"] = {k: m.name for k, m in self.modalities.items()}
        serialized["checksum"] = self._compute_checksum(new_table, self.modalities)

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
            ext = fs.get_extension(value)
            dst = fs.join(self.cache_dir, column, f"{h}.{ext}")

            # The reason for caching the path is to speed-up retrieval. Hashing can be slow and with large
            # datasets this could become a bottleneck.
            self._path_to_hash[column][value] = dst

        return self._path_to_hash[column][value]

    def get_data(self, row, col):
        """
        Helper function to get data from the Dataset. Since the dataset might contain pointers to
        external files, data retrieval is more complicated than just indexing the table.
        """

        def _load(path: str) -> np.ndarray:
            # NOTE (cwognum): We will probably want to support more formats than just .zarr in the future
            return zarr.convenience.load(path)

        value = self.table.loc[row, col]
        if not self.modalities[col].is_pointer():
            return value

        # In the case it is a pointer column, we need to load additional data into memory
        # We first check if the data has been downloaded to the cache.
        path = self.get_cache_path(column=col, value=value)
        if fs.exists(path):
            return _load(path)

        # If it doesn't exist, we load from the original path and warn if not local
        if not fs.is_local_path(path) and not self._has_been_warned:
            logger.warning(
                f"You're loading data from a remote location. "
                f"To speed up this process, consider caching the dataset first "
                f"using {self.__class__.__name__}.download()"
            )
            self._has_been_warned = True
        return _load(value)

    def size(self):
        return len(self), len(self.table.columns)

    def download(self, column_subset: Optional[Union[str, List[str]]] = None) -> str:
        """
        Download all additional data for pointer columns.
        """

        if column_subset is None:
            column_subset = self.table.columns.values.tolist()
        if not isinstance(column_subset, list):
            column_subset = [column_subset]

        for col in column_subset:
            if not self.modalities[col].is_pointer():
                continue

            paths = self.table[col].values
            for src in paths:
                dst = self.get_cache_path(col, src)
                robust_file_copy(src, dst)

        return self.cache_dir
