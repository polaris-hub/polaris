import logging
import os
import re
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import zarr
from pydantic import (
    PrivateAttr,
    computed_field,
    model_validator,
)

from polaris.utils.zarr._manifest import generate_zarr_manifest, calculate_file_md5
from polaris.evaluate import ResultsMetadataV2
from polaris.evaluate._predictions import BenchmarkPredictions
from polaris.benchmark import BenchmarkV2Specification

if TYPE_CHECKING:
    from polaris.benchmark import BenchmarkV2Specification

logger = logging.getLogger(__name__)


class BenchmarkPredictionsV2(BenchmarkPredictions, ResultsMetadataV2):
    """
    Prediction artifact for uploading predictions to a Benchmark V2.
    Stores predictions as a Zarr archive, with manifest and metadata for reproducibility and integrity.
    In addition to the predictions data, it contains metadata that describes how these predictions
    were generated, including the model used and contributors involved.

    Attributes:
        Inherits from BenchmarkPredictions and ResultsMetadataV2.
    """

    benchmark: BenchmarkV2Specification
    _artifact_type = "prediction"
    _zarr_root_path: str | None = PrivateAttr(None)
    _zarr_manifest_path: str | None = PrivateAttr(None)
    _zarr_manifest_md5sum: str | None = PrivateAttr(None)
    _zarr_root: zarr.Group | None = PrivateAttr(None)
    _temp_dir: str | None = PrivateAttr(None)

    @model_validator(mode="after")
    def check_prediction_dtypes(self):
        dataset_root = self.benchmark.dataset.zarr_root
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
        root = self.zarr_root
        dataset_root = self.benchmark.dataset.zarr_root

        for test_set_label, test_set_predictions in self.predictions.items():
            # Create a group for each test set
            test_set_group = root.require_group(test_set_label)
            for col in self.benchmark.target_cols:
                data = test_set_predictions[col]
                template = dataset_root[col]
                test_set_group.array(
                    name=col,
                    data=data,
                    dtype=template.dtype,
                    compressor=template.compressor,
                    filters=template.filters,
                    chunks=template.chunks,
                    object_codec=getattr(template, "object_codec", None),
                    overwrite=True,
                )

        return Path(self.zarr_root_path)

    @property
    def zarr_root(self) -> zarr.Group:
        """Get the zarr Group object corresponding to the root, creating it if it doesn't exist."""
        if self._zarr_root is None:
            store = zarr.DirectoryStore(self.zarr_root_path)
            self._zarr_root = zarr.group(store=store)
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

    @computed_field
    @property
    def benchmark_artifact_id(self) -> str:
        return self.benchmark.artifact_id

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
        if self._temp_dir and os.path.exists(self._temp_dir):
            os.remove(self._temp_dir)
