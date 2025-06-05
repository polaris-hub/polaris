import logging
import re
import os

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
from polaris.benchmark import BenchmarkV2Specification
from polaris.dataset.zarr.codecs import RDKitMolCodec, AtomArrayCodec

logger = logging.getLogger(__name__)


class Predictions(BaseArtifactModel):
    """
    Prediction artifact for uploading predictions to a Benchmark V2.
    Stores predictions as a Zarr archive, with manifest and metadata for reproducibility and integrity.

    Attributes:
        benchmark: The BenchmarkV2Specification instance this prediction is for.
        model: (Optional) The Model artifact used to generate these predictions.
        predictions: A dictionary mapping column names to prediction arrays/lists.
        zarr_root_path: Required path to a Zarr archive containing the predictions.
    """

    _artifact_type = "prediction"
    benchmark: BenchmarkV2Specification
    model: Model | None = Field(None, exclude=True)
    predictions: dict[str, list | np.ndarray | list[object]] = Field(exclude=True)
    zarr_root_path: str

    _zarr_manifest_path: str | None = PrivateAttr(None)
    _zarr_manifest_md5sum: str | None = PrivateAttr(None)
    _zarr_root: zarr.Group | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _initialize_zarr_from_predictions(self) -> Self:
        """Initialize Zarr archive from predictions."""
        # Create the Zarr archive with predictions at the specified zarr_root_path
        self._create_zarr_from_predictions()

        return self

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
            root.create_array(col, data=data, **codec_kwargs)

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
        store = zarr.DirectoryStore(self.zarr_root_path)
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
            # Use the parent directory of the zarr root as the output directory
            output_dir = os.path.dirname(self.zarr_root_path)
            zarr_manifest_path = generate_zarr_manifest(self.zarr_root_path, output_dir)
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
