import numpy as np
import zarr
import dataclasses
import pandas as pd
import datamol as dm
from hashlib import sha256
from typing import Dict, Optional, Union, List

from loguru import logger

from polaris.dataset import Modality
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.errors import InvalidDatasetError


@dataclasses.dataclass
class DatasetInfo:
    """
    Stores additional information about the dataset.
    NOTE (cwognum): Currently implemented as a separate class because it is saved in a separate file.
        Logically made sense to me, but happy to discuss alternatives.
    """

    # The public-facing name of the dataset
    name: str

    # A beginner-friendly description of the dataset
    description: str

    # The data source, e.g. a DOI, Github repo or URL
    source: str

    # Annotates each column with the modality of that column
    modalities: Dict[str, Modality]

    # Version of the dataset
    version: Optional[str] = None

    def serialize(self) -> dict:
        """Convert the object into a YAML-serializable dictionary."""
        data = dataclasses.asdict(self)
        data["modalities"] = {k: v.name for k, v in data["modalities"].items()}
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DatasetInfo":
        """Takes a dictionary and converts it back into a DatasetInfo object."""
        data["modalities"] = {k: Modality[v] for k, v in data["modalities"].items()}
        return cls(**data)


class Dataset:
    """
    A dataset class is a wrapper around a pandas DataFrame.

    It is a tabular format which contains all information to construct an ML-dataset, but is not ready yet.
    For example, if the dataset contains the Image modality, the image data is not yet loaded but the column
    rather contains a pointer to the image file.

    A Dataset can have multiple modalities or targets, can be sparse and can be part of one or multiple Benchmarks.
    """

    _SUPPORTED_POINTER_EXTENSIONS = ["zarr"]

    def __init__(self, table: pd.DataFrame, info: DatasetInfo, cache_dir: Optional[str] = None):
        self.table = table
        self.info = info
        self._cache_dir = cache_dir
        self._path_to_hash = {}
        self._has_been_warned = False

        for c in table.columns:
            # Fill missing modalities
            if c not in info.modalities:
                info.modalities[c] = Modality.UNKNOWN

            # Check pointers
            if info.modalities[c].is_pointer():
                self._path_to_hash[c] = {}
                paths = table[c].values
                if any(dm.fs.get_extension(p) not in self._SUPPORTED_POINTER_EXTENSIONS for p in paths):
                    raise InvalidDatasetError(
                        f"The {c} [{info.modalities[c]}] column contains pointers with invalid extensions. "
                        f"Choose from {self._SUPPORTED_POINTER_EXTENSIONS}"
                    )

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        s = (
            f"Name: {self.info.name}\n"
            f"Description: {self.info.description}\n"
            f"Source: {self.info.source}\n"
        )
        self.table.loc[""] = [f"[{self.info.modalities[c]}]" for c in self.table.columns]
        s += repr(self.table)
        self.table.drop(self.table.tail(1).index, inplace=True)
        return s

    @property
    def cache_dir(self) -> str:
        if self._cache_dir is None:
            self._cache_dir = dm.fs.join(DEFAULT_CACHE_DIR, self.info.name, self.info.version)
        return self._cache_dir

    def get_cache_path(self, column: str, value: str) -> str:
        """
        Returns where the data _would be_ cached for any entry in the pointer columns
        """
        if value not in self._path_to_hash[column]:
            h = sha256(value.encode("utf-8")).hexdigest()
            ext = dm.fs.get_extension(value)
            dst = dm.fs.join(self.cache_dir, column, f"{h}.{ext}")

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
            # NOTE (cwognum): For now we assume external data always uses the .zarr format
            #   Ultimately, we will probably want to extend this method into something more flexible.
            assert dm.fs.get_extension(path) in self._SUPPORTED_POINTER_EXTENSIONS

            # TODO (cwognum): I have not gone through the Zarr docs in detail and am sure this can be improved.
            return zarr.convenience.load(path)

        value = self.table.loc[row, col]
        if not self.info.modalities[col].is_pointer():
            return value

        # In the case it is a pointer column, we need to load additional data into memory
        # We first check if the data has been downloaded to the cache.
        path = self.get_cache_path(column=col, value=value)
        if dm.fs.exists(path):
            return _load(path)

        # If it doesn't exist, we load from the original path and warn if not local
        if not dm.fs.is_local_path(path) and not self._has_been_warned:
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
        TODO (cwognum): This now expects a file per row. Does that make sense with .zarr?
        """

        if column_subset is None:
            column_subset = self.table.columns.values.tolist()
        if not isinstance(column_subset, list):
            column_subset = [column_subset]

        for col in column_subset:
            if not self.info.modalities[col].is_pointer():
                continue

            paths = self.table[col].values
            for src in paths:
                dst = self.get_cache_path(col, src)

                # TODO (cwognum): A retry mechanism
                if not dm.fs.exists(dst):
                    dm.fs.copy_dir(src, dst)

        return self.cache_dir
