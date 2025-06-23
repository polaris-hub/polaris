import logging
import os
import re
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import zarr
from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
    model_validator,
)

from polaris.utils.zarr._manifest import generate_zarr_manifest, calculate_file_md5
from polaris.evaluate import ResultsMetadataV2
from polaris.evaluate._predictions import BenchmarkPredictions
from polaris.dataset import Subset

if TYPE_CHECKING:
    from polaris.benchmark import BenchmarkV2Specification

logger = logging.getLogger(__name__)


class BenchmarkPredictionsV2(ResultsMetadataV2):
    """
    Prediction artifact for uploading predictions to a Benchmark V2.
    Stores predictions as a Zarr archive, with manifest and metadata for reproducibility and integrity.
    In addition to the predictions data, it contains metadata that describes how these predictions
    were generated, including the model used and contributors involved.

    Attributes:
        benchmark: The BenchmarkV2Specification instance this prediction is for.
        model: (Optional) The Model artifact used to generate these predictions.
        predictions: A dictionary mapping column names to prediction arrays/lists.
    """

    _artifact_type = "prediction"
    benchmark: "BenchmarkV2Specification"
    predictions: dict[str, list | np.ndarray | list[object]] = Field(exclude=True)

    _zarr_root_path: str | None = PrivateAttr(None)
    _zarr_manifest_path: str | None = PrivateAttr(None)
    _zarr_manifest_md5sum: str | None = PrivateAttr(None)
    _zarr_root: zarr.Group | None = PrivateAttr(None)
    _temp_dir: str | None = PrivateAttr(None)

    @model_validator(mode="before")
    @classmethod
    def _validate_predictions(cls, data: dict) -> dict:
        """Validate and standardize the predictions format."""
        if "predictions" not in data or "benchmark" not in data:
            raise ValueError("Both 'predictions' and 'benchmark' fields are required")

        benchmark = data["benchmark"]
        benchmark = BenchmarkV2Specification(**benchmark)
        dataset_root = benchmark.dataset.zarr_root

        target_cols = list(benchmark.target_cols)
        test_set_labels = (
            list(benchmark.test_set_labels) if hasattr(benchmark, "test_set_labels") else ["test"]
        )

        # Get test set sizes from the benchmark's test split
        _, test_splits = benchmark.get_train_test_split()
        if isinstance(test_splits, Subset):
            test_set_sizes = {test_set_labels[0]: len(test_splits)}
        else:
            test_set_sizes = {label: len(split) for label, split in test_splits.items()}

        temp_data = {
            "predictions": data["predictions"],
            "target_labels": target_cols,
            "test_set_labels": test_set_labels,
            "test_set_sizes": test_set_sizes,
        }

        normalized_data = BenchmarkPredictions._validate_predictions(temp_data)
        normalized_predictions = normalized_data["predictions"]

        processed_predictions = {}

        for test_set_label, test_set_predictions in normalized_predictions.items():
            processed_predictions[test_set_label] = {}

            for col, preds in test_set_predictions.items():
                dataset_array = dataset_root[col]

                if dataset_array.dtype == np.dtype(object):
                    arr = np.empty(len(preds), dtype=object)
                    for i, item in enumerate(preds):
                        arr[i] = item
                else:
                    arr = np.asarray(preds, dtype=dataset_array.dtype)

                processed_predictions[test_set_label][col] = arr

        data["predictions"] = processed_predictions
        return data

    def to_zarr(self) -> Path:
        """Create a Zarr archive from the predictions dictionary.

        This method should be called explicitly when ready to write predictions to disk.
        """
        root = self.zarr_root
        dataset_root = self.benchmark.dataset.zarr_root

        # Copy each array exactly
        for col in self.benchmark.target_cols:
            data = self.predictions[col]
            template = dataset_root[col]
            root.array(
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
