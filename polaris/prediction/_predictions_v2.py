import logging
import os
import re
import shutil
from pathlib import Path
import tempfile
from enum import Enum

import numpy as np
import zarr
from pydantic import (
    PrivateAttr,
    Field,
    model_validator,
)
from numcodecs import MsgPack, VLenBytes
from fastpdb import struc
from rdkit import Chem

from polaris.utils.zarr._manifest import generate_zarr_manifest, calculate_file_md5
from polaris.utils.zarr.codecs import (
    convert_dict_to_atomarray,
    convert_bytes_to_mol,
    convert_atomarray_to_dict,
    convert_mol_to_bytes,
)
from polaris.evaluate import ResultsMetadataV2
from polaris.evaluate._predictions import BenchmarkPredictions

logger = logging.getLogger(__name__)

# Reserved metadata key for storing original Python type
RESERVED_TYPE_KEY = "python_type"


class ReservedTypes(str, Enum):
    """Reserved type identifiers for object data stored in Zarr arrays."""

    RDKIT_MOL = "rdkit.Chem.Mol"
    ATOM_ARRAY = "biotite.structure.AtomArray"


class BenchmarkPredictionsV2(BenchmarkPredictions, ResultsMetadataV2):
    """
    Prediction artifact for uploading predictions to a Benchmark V2.
    Stores predictions as a Zarr archive, with manifest and metadata for reproducibility and integrity.
    In addition to the predictions data, it contains metadata that describes how these predictions
    were generated, including the model used and contributors involved.

    Attributes:
        dataset_zarr_root: The zarr root of the dataset, used for dtype validation and as template for zarr arrays.
        benchmark_artifact_id: The artifact ID of the benchmark these predictions are for.

    For additional metadata attributes, see the base classes.
    """

    predictions: dict = Field(exclude=True)  # NumPy arrays cannot be JSON serialized
    dataset_zarr_root: zarr.Group = Field(exclude=True)  # Zarr Group cannot be JSON serialized
    benchmark_artifact_id: str
    _artifact_type = "prediction"
    _zarr_root_path: str | None = PrivateAttr(None)
    _zarr_manifest_path: str | None = PrivateAttr(None)
    _zarr_manifest_md5sum: str | None = PrivateAttr(None)
    _zarr_root: zarr.Group | None = PrivateAttr(None)
    _temp_dir: str | None = PrivateAttr(None)

    @model_validator(mode="after")
    def check_prediction_dtypes(self):
        dataset_root = self.dataset_zarr_root
        for test_set_label, test_set_predictions in self.predictions.items():
            for col, preds in test_set_predictions.items():
                dataset_array = dataset_root[col]
                arr = np.asarray(preds)
                if arr.dtype != dataset_array.dtype:
                    raise ValueError(
                        f"Dtype mismatch for column '{col}' in test set '{test_set_label}': "
                        f"predictions dtype {arr.dtype} != dataset dtype {dataset_array.dtype}"
                    )
        return self

    def to_zarr(self) -> Path:
        """Create a Zarr archive from the predictions dictionary.

        This method should be called explicitly when ready to write predictions to disk.
        """
        # Get zarr root for writing
        store = zarr.DirectoryStore(self.zarr_root_path)
        root = zarr.group(store=store)
        dataset_root = self.dataset_zarr_root

        for test_set_label, test_set_predictions in self.predictions.items():
            # Create a group for each test set
            test_set_group = root.require_group(test_set_label)
            for col in self.target_labels:
                data = test_set_predictions[col]
                template = dataset_root[col]

                # Handle object data conversion
                if template.dtype == object:
                    sample = next((item for item in data if item is not None), None)

                    if isinstance(sample, Chem.Mol):
                        codec = VLenBytes()
                        final_data = [convert_mol_to_bytes(item) for item in data]
                        filters = [codec]
                        attributes = {RESERVED_TYPE_KEY: ReservedTypes.RDKIT_MOL}
                    elif isinstance(sample, struc.AtomArray):
                        codec = MsgPack()
                        final_data = [convert_atomarray_to_dict(item) for item in data]
                        filters = [codec]
                        attributes = {RESERVED_TYPE_KEY: ReservedTypes.ATOM_ARRAY}
                    else:
                        # Fall back to dataset template for unknown types
                        final_data = list(data)
                        filters = template.filters
                        attributes = {}
                else:
                    # Non-object data uses original data and template filters
                    final_data = data
                    filters = template.filters
                    attributes = {}

                # Single array creation for both cases
                zarr_array = test_set_group.array(
                    name=col,
                    data=final_data,
                    dtype=template.dtype,
                    compressor=template.compressor,
                    filters=filters,
                    chunks=template.chunks,
                    overwrite=True,
                )

                # Set attributes after creation if we have any
                if attributes:
                    zarr_array.attrs.update(attributes)

        return Path(self.zarr_root_path)

    def get_converted_predictions(self) -> dict:
        """Get all predictions with automatic conversion back to original object types.

        Returns:
            Full predictions dictionary with same structure as self.predictions,
            but with object data converted back to original types (AtomArray, RDKit Mol)
        """
        converted_predictions = {}

        for test_set_label in self.test_set_labels:
            test_set_group = self.zarr_root.require_group(test_set_label)
            converted_predictions[test_set_label] = {}

            for target in self.target_labels:
                zarr_array = test_set_group[target]
                data = zarr_array[:]

                # Check if this needs conversion (object data)
                template = self.dataset_zarr_root[target]
                if template.dtype == object:
                    # Use metadata to determine conversion type
                    python_type = zarr_array.attrs.get(RESERVED_TYPE_KEY)
                    if python_type == ReservedTypes.RDKIT_MOL:
                        converter = convert_bytes_to_mol
                    elif python_type == ReservedTypes.ATOM_ARRAY:
                        converter = convert_dict_to_atomarray
                    else:
                        converter = None

                    # Apply conversion
                    if converter:
                        converted = [converter(item) for item in data]
                        converted_predictions[test_set_label][target] = np.array(converted, dtype=object)
                    else:
                        converted_predictions[test_set_label][target] = data
                else:
                    # Non-object data, use as-is
                    converted_predictions[test_set_label][target] = data

        return converted_predictions

    @property
    def zarr_root(self) -> zarr.Group:
        """Get the zarr Group object corresponding to the root, creating it if it doesn't exist."""
        if self._zarr_root is None:
            store = zarr.DirectoryStore(self.zarr_root_path)
            root = zarr.group(store=store)
            self._zarr_root = root
        return self._zarr_root

    @property
    def zarr_root_path(self) -> str:
        """Get the path to the Zarr archive root."""
        if self._zarr_root_path is None:
            # Create a temporary directory if not already set
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(prefix="polaris_predictions_")
            self._zarr_root_path = str(Path(self._temp_dir) / "predictions.zarr")
        return self._zarr_root_path

    @property
    def columns(self):
        return list(self.zarr_root.keys())

    @property
    def n_rows(self):
        cols = self.columns
        if not cols:
            raise ValueError("No columns found in predictions archive.")
        example = self.zarr_root[cols[0]]
        return len(example)

    @property
    def rows(self):
        return range(self.n_rows)

    @property
    def zarr_manifest_path(self):
        if self._zarr_manifest_path is None:
            # Use the temp directory as the output directory
            zarr_manifest_path = generate_zarr_manifest(self.zarr_root_path, self._temp_dir)
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

    def __repr__(self):
        return self.model_dump_json(by_alias=True, indent=2)

    def __str__(self):
        return self.__repr__()

    def __del__(self) -> None:
        if hasattr(self, "_temp_dir") and self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
