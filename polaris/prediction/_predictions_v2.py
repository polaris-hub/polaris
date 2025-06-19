import logging
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
from typing_extensions import Self

from polaris.utils.types import IncomingPredictionsType
from polaris.utils.zarr._manifest import generate_zarr_manifest, calculate_file_md5
from polaris.evaluate import ResultsMetadataV2
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from polaris.benchmark import BenchmarkV2Specification

logger = logging.getLogger(__name__)


class Predictions(ResultsMetadataV2):
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
            return data

        # Ensure the type of the incoming predictions is correct
        validator = TypeAdapter(IncomingPredictionsType, config={"arbitrary_types_allowed": True})
        predictions = validator.validate_python(data["predictions"])

        # Get benchmark and dataset info
        benchmark = data["benchmark"]
        dataset_root = benchmark.dataset.zarr_root

        # Get expected size from test split
        _, test = benchmark.get_train_test_split()
        expected_size = len(test)

        # Validate predictions against benchmark target columns
        target_cols = list(benchmark.target_cols)
        if not isinstance(predictions, dict) or set(predictions.keys()) != set(target_cols):
            raise ValueError(
                "The predictions should be a dictionary with the target columns as keys. "
                f"Expected columns: {target_cols}, got: {list(predictions.keys()) if isinstance(predictions, dict) else type(predictions)}"
            )

        # Process each column's predictions
        processed_predictions = {}
        for col, preds in predictions.items():
            # Get the array configuration from the dataset
            dataset_array = dataset_root[col]

            # Validate length
            if len(preds) != expected_size:
                raise ValueError(
                    f"Predictions size mismatch: Column '{col}' has {len(preds)} predictions, "
                    f"but test set has size {expected_size}"
                )

            # Convert to array matching dataset's dtype
            if dataset_array.dtype == np.dtype(object):
                arr = np.empty(len(preds), dtype=object)
                for i, item in enumerate(preds):
                    arr[i] = item
            else:
                arr = np.asarray(preds, dtype=dataset_array.dtype)

            processed_predictions[col] = arr

        data["predictions"] = processed_predictions
        return data

    @model_validator(mode="after")
    def _create_zarr_from_predictions(self) -> Self:
        """Create a Zarr archive from the predictions dictionary."""
        root = self.zarr_root
        dataset_root = self.benchmark.dataset.zarr_root

        for col, data in self.predictions.items():
            # Copy array configuration from dataset
            dataset_array = dataset_root[col]
            root.array(col, data=data, **dataset_array.attrs.asdict())

        return self

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
