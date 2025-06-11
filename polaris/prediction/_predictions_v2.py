import logging
import re
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import zarr
from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
    model_validator,
)
from typing_extensions import Self

from polaris._artifact import BaseArtifactModel
from polaris.model import Model
from polaris.utils.types import (
    HubOwner,
)
from polaris.dataset.zarr._manifest import generate_zarr_manifest, calculate_file_md5
from polaris.dataset.zarr import MemoryMappedDirectoryStore
from polaris.benchmark import BenchmarkV2Specification
from polaris.dataset.zarr.codecs import RDKitMolCodec, AtomArrayCodec
from polaris.utils.constants import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

_CACHE_SUBDIR = "predictions"


class Predictions(BaseArtifactModel):
    """
    Prediction artifact for uploading predictions to a Benchmark V2.
    Stores predictions as a Zarr archive, with manifest and metadata for reproducibility and integrity.

    Attributes:
        benchmark: The BenchmarkV2Specification instance this prediction is for.
        model: (Optional) The Model artifact used to generate these predictions.
        predictions: A dictionary mapping column names to prediction arrays/lists.
    """

    _artifact_type = "prediction"
    benchmark: BenchmarkV2Specification
    model: Model | None = Field(None, exclude=True)
    predictions: dict[str, list | np.ndarray | list[object]] = Field(exclude=True)

    _zarr_root_path: str | None = PrivateAttr(None)
    _zarr_manifest_path: str | None = PrivateAttr(None)
    _zarr_manifest_md5sum: str | None = PrivateAttr(None)
    _zarr_root: zarr.Group | None = PrivateAttr(None)
    _cache_dir: str | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _initialize_cache_and_zarr(self) -> Self:
        """Set up the cache directory and initialize Zarr archive from predictions."""
        # Set up cache dir if not already set
        if self._cache_dir is None:
            cache_root = Path(DEFAULT_CACHE_DIR) / _CACHE_SUBDIR
            cache_root.mkdir(parents=True, exist_ok=True)
            self._cache_dir = str(cache_root / str(uuid4()))
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        # Set zarr_root_path if not already set
        if self._zarr_root_path is None:
            self._zarr_root_path = str(Path(self._cache_dir) / "predictions.zarr")
        # Create the Zarr archive with predictions at the specified zarr_root_path
        self._create_zarr_from_predictions()
        return self

    @property
    def zarr_root_path(self) -> str:
        return self._zarr_root_path

    def _create_zarr_from_predictions(self):
        """Create a Zarr archive from the predictions dictionary."""
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.zarr_root_path), exist_ok=True)

        store = zarr.DirectoryStore(self.zarr_root_path)
        root = zarr.group(store=store)

        # Get target columns from benchmark
        target_cols = list(self.benchmark.target_cols)

        for col in target_cols:
            if col not in self.predictions:
                raise ValueError(f"Predictions for target column '{col}' not provided")

            data = self.predictions[col]

            # Determine the appropriate codec based on the dataset column annotation
            codec_kwargs = {}
            if hasattr(self.benchmark.dataset, "annotations") and col in self.benchmark.dataset.annotations:
                annotation = self.benchmark.dataset.annotations[col]

                # Check if this column contains complex objects that need special codecs
                if len(data) > 0:
                    # Extract a sample value to determine the type
                    if isinstance(data, (list, tuple)):
                        sample_value = data[0]
                    elif hasattr(data, "__getitem__"):
                        try:
                            sample_value = data[0]
                        except (IndexError, TypeError):
                            sample_value = None
                    else:
                        sample_value = None

                    if sample_value is not None:
                        sample_type_name = type(sample_value).__name__
                        if sample_type_name == "Mol" and "rdkit" in str(type(sample_value).__module__):
                            codec_kwargs["object_codec"] = RDKitMolCodec()
                            codec_kwargs["dtype"] = object
                        elif sample_type_name == "AtomArray" and "struc" in str(
                            type(sample_value).__module__
                        ):
                            codec_kwargs["object_codec"] = AtomArrayCodec()
                            codec_kwargs["dtype"] = object
                        elif annotation.dtype == np.dtype(object):
                            # For other object types, use object dtype
                            codec_kwargs["dtype"] = object

            # Create the array in the Zarr archive
            if "object_codec" in codec_kwargs:
                # For object codecs, we need to create a numpy object array first
                # Use np.empty to avoid numpy trying to convert AtomArrays to numpy arrays
                data_array = np.empty(len(data), dtype=object)
                for i, item in enumerate(data):
                    data_array[i] = item
                root.array(col, data=data_array, **codec_kwargs)
            else:
                root.array(col, data=data, **codec_kwargs)

    @computed_field
    @property
    def benchmark_artifact_id(self) -> str:
        return self.benchmark.artifact_id if self.benchmark else None

    @computed_field
    @property
    def model_artifact_id(self) -> str:
        return self.model.artifact_id if self.model else None

    @property
    def zarr_root(self) -> zarr.Group:
        """Get the zarr Group object corresponding to the root."""
        if self._zarr_root is not None:
            return self._zarr_root

        # The Zarr archive was created during initialization from predictions
        store = MemoryMappedDirectoryStore(self.zarr_root_path)
        self._zarr_root = zarr.open_group(store, mode="r+")
        return self._zarr_root

    @property
    def columns(self):
        return list(self.zarr_root.keys())

    @property
    def n_rows(self):
        cols = self.columns
        if not cols:
            raise ValueError("No columns found in predictions archive.")
        example = self.zarr_root[cols[0]]
        if isinstance(example, zarr.Group):
            return len(example)
        return len(example)

    @property
    def rows(self):
        return range(self.n_rows)

    @property
    def zarr_manifest_path(self):
        if self._zarr_manifest_path is None:
            # Use the cache directory as the output directory
            zarr_manifest_path = generate_zarr_manifest(self.zarr_root_path, self._cache_dir)
            self._zarr_manifest_path = zarr_manifest_path
        return self._zarr_manifest_path

    @property
    def zarr_manifest_md5sum(self):
        if not self.has_zarr_manifest_md5sum:
            logger.info("Computing the checksum. This can be slow for large predictions archives.")
            self.zarr_manifest_md5sum = calculate_file_md5(self.zarr_manifest_path)
        return self._zarr_manifest_md5sum

    @zarr_manifest_md5sum.setter
    def zarr_manifest_md5sum(self, value: str):
        if not re.fullmatch(r"^[a-f0-9]{32}$", value):
            raise ValueError("The checksum should be the 32-character hexdigest of a 128 bit MD5 hash.")
        self._zarr_manifest_md5sum = value

    @property
    def has_zarr_manifest_md5sum(self):
        return self._zarr_manifest_md5sum is not None

    def upload_to_hub(
        self,
        owner: HubOwner | str | None = None,
        **kwargs: dict,
    ):
        """
        Wrapper around PolarisHubClient.upload_predictions
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(**kwargs) as client:
            return client.upload_predictions(self, owner=owner)

    def __repr__(self):
        return self.model_dump_json(by_alias=True, indent=2)

    def __str__(self):
        return self.__repr__()
