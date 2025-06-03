import logging
import re
from typing import Any

import numpy as np
import zarr
from pydantic import model_validator
from pydantic import TypeAdapter

from polaris._artifact import BaseArtifactModel
from polaris.model import Model
from polaris.utils.types import HubOwner
from polaris.dataset.zarr._manifest import generate_zarr_manifest, calculate_file_md5
from polaris.hub.client import PolarisHubClient

logger = logging.getLogger(__name__)

_INDEX_ARRAY_KEY = "__index__"


class BenchmarkPredictions:
    """
    Base model to represent predictions in the Polaris code base.

    Guided by [Postel's Law](https://en.wikipedia.org/wiki/Robustness_principle),
    this class normalizes different formats to a single, internal representation.

    Attributes:
        predictions: The predictions for the benchmark.
        target_labels: The target columns for the associated benchmark.
        test_set_labels: The names of the test sets for the associated benchmark.
        test_set_sizes: The number of rows in each test set for the associated benchmark.
    """

    predictions: dict
    target_labels: list[str]
    test_set_labels: list[str]
    test_set_sizes: dict[str, int]

    def __init__(self, predictions, target_labels, test_set_labels, test_set_sizes):
        self.predictions = predictions
        self.target_labels = target_labels
        self.test_set_labels = test_set_labels
        self.test_set_sizes = test_set_sizes

    def _serialize_predictions(self, predictions):
        """
        Recursively converts all numpy values in the predictions dictionary to lists
        so they can be serialized.
        """

        def convert_to_list(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, dict):
                return {k: convert_to_list(v) for k, v in v.items()}

        return convert_to_list(predictions)

    def _validate_labels(self, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("The predictions contain duplicate columns")
        return v

    def _validate_predictions(self, data: dict) -> dict:
        """Normalizes the predictions format to a standard representation we use internally"""

        # This model validator runs before any Pydantic internal validation.
        # This way we can normalize the incoming data to a standard representation.
        # However, this implies that the fields can theoretically be any type.

        # Ensure the type of the incoming predictions is correct
        predictions = data.get("predictions")

        # Ensure the type of the target_labels and test_set_labels is correct
        target_labels = data.get("target_labels")
        test_set_labels = data.get("test_set_labels")

        validator = TypeAdapter(dict[str, int])
        test_set_sizes = validator.validate_python(data.get("test_set_sizes"))

        # Normalize the predictions to a standard representation
        predictions = self._normalize_predictions(predictions, target_labels, test_set_labels)

        # Update class data with the normalized fields. Use of the `update()` method
        # is required to prevent overwriting class data when this class is inherited.
        data.update(
            {
                "predictions": predictions,
                "target_labels": target_labels,
                "test_set_labels": test_set_labels,
                "test_set_sizes": test_set_sizes,
            }
        )

        return data

    def check_test_set_size(self):
        """Verify that the size of all predictions"""
        for test_set_label, test_set in self.predictions.items():
            for target in test_set.values():
                if test_set_label not in self.test_set_sizes:
                    raise ValueError(f"Expected size for test set '{test_set_label}' is not defined")

                if len(target) != self.test_set_sizes[test_set_label]:
                    raise ValueError(
                        f"Predictions size mismatch: The predictions for test set '{test_set_label}' "
                        f"should have a size of {self.test_set_sizes[test_set_label]}, but have a size of {len(target)}."
                    )
        return self

    def _normalize_predictions(self, predictions, target_labels, test_set_labels):
        """
        Normalizes  the predictions to a standard representation we use internally.
        This standard representation is a nested, two-level dictionary:
        `{test_set_name: {target_column: np.ndarray}}`
        """
        # (1) If the predictions are already fully specified, no need to do anything
        if self._is_fully_specified(predictions, target_labels, test_set_labels):
            return predictions

        # If not fully specified, we distinguish 4 cases based on the type of benchmark.
        is_single_task = len(target_labels) == 1
        is_single_test = len(test_set_labels) == 1

        # (2) Single-task, single test set: We expect a numpy array as input.
        if is_single_task and is_single_test:
            if isinstance(predictions, dict):
                raise ValueError(
                    "The predictions for single-task, single test set benchmarks should be a numpy array."
                )

            predictions = {test_set_labels[0]: {target_labels[0]: predictions}}

        # (3) Single-task, multiple test sets: We expect a dictionary with the test set labels as keys.
        elif is_single_task and not is_single_test:
            if not isinstance(predictions, dict) or set(predictions.keys()) != set(test_set_labels):
                raise ValueError(
                    "The predictions for single-task, multiple test sets benchmarks "
                    "should be a dictionary with the test set labels as keys."
                )
            predictions = {k: {target_labels[0]: v} for k, v in predictions.items()}

        # (4) Multi-task, single test set: We expect a dictionary with the target labels as keys.
        elif not is_single_task and is_single_test:
            if not isinstance(predictions, dict) or set(predictions.keys()) != set(target_labels):
                raise ValueError(
                    "The predictions for multi-task, single test set benchmarks "
                    "should be a dictionary with the target labels as keys."
                )
            predictions = {test_set_labels[0]: predictions}

        # (5) Multi-task, multi-test sets: The predictions should be fully-specified
        else:
            raise ValueError(
                "The predictions for multi-task, multi-test sets benchmarks should be fully-specified "
                "as a nested, two-level dictionary: { test_set_name: { target_column: np.ndarray } }"
            )

        return predictions

    def _is_fully_specified(self, predictions, target_labels, test_set_labels):
        """
        Check if the predictions are fully specified for the target columns and test set names.
        """
        # Not a dictionary
        if not isinstance(predictions, dict):
            return False

        # Outer-level of the dictionary should correspond to the test set names
        if set(predictions.keys()) != set(test_set_labels):
            return False

        # Inner-level of the dictionary should correspond to the target columns
        for test_set_predictions in predictions.values():
            if not isinstance(test_set_predictions, dict):
                return False
            if set(test_set_predictions.keys()) != set(target_labels):
                return False

        return True

    def get_subset(self, test_set_subset: list[str] | None = None, target_subset: list[str] | None = None):
        """Return a subset of the original predictions"""

        if test_set_subset is None and target_subset is None:
            return self

        if test_set_subset is None:
            test_set_subset = self.test_set_labels
        if target_subset is None:
            target_subset = self.target_labels

        predictions = {}
        for test_set_name in self.predictions.keys():
            if test_set_name not in test_set_subset:
                continue

            for target_name, target in self.predictions[test_set_name].items():
                if target_name not in target_subset:
                    continue

                predictions[test_set_name] = {target_name: target}

        test_set_sizes = {k: v for k, v in self.test_set_sizes.items() if k in predictions.keys()}
        return BenchmarkPredictions(
            predictions=predictions,
            target_labels=target_subset,
            test_set_labels=test_set_subset,
            test_set_sizes=test_set_sizes,
        )

    def get_size(self, test_set_subset: list[str] | None = None, target_subset: list[str] | None = None):
        """Return the total number of predictions, allowing for filtering by test set and target"""
        if test_set_subset is None and target_subset is None:
            return sum(self.test_set_sizes.values()) * len(self.target_labels)
        return len(self.get_subset(test_set_subset, target_subset))

    def flatten(self):
        """Return the predictions as a single, flat numpy array"""
        if len(self.test_set_labels) != 1 and len(self.target_labels) != 1:
            raise ValueError(
                "Can only flatten predictions for benchmarks with a single test set and a single target"
            )
        return self.predictions[self.test_set_labels[0]][self.target_labels[0]]

    def __len__(self):
        """Return the total number of predictions"""
        return self.get_size()


class CompetitionPredictions(BenchmarkPredictions):
    """
    Predictions for competition benchmarks.

    This object is to be used as input to
    [`PolarisHubClient.submit_competition_predictions`][polaris.hub.client.PolarisHubClient.submit_competition_predictions].
    It is used to ensure that the structure of the predictions are compatible with evaluation methods on the Polaris Hub.
    In addition to the predictions, it contains metadata that describes a predictions object.

    Attributes:
        name: A slug-compatible name for the artifact. It is redeclared here to be required.
        owner: A slug-compatible name for the owner of the artifact. It is redeclared here to be required.
        report_url: A URL to a report/paper/write-up which describes the methods used to generate the predictions.
    """

    _artifact_type = "competition-prediction"

    name: str
    owner: str
    paper_url: str

    def __repr__(self):
        return self.model_dump_json(by_alias=True, indent=2)

    def __str__(self):
        return self.__repr__()


class Predictions(BaseArtifactModel):
    """
    Prediction artifact for uploading predictions to a Benchmark V2.
    Stores predictions as a Zarr archive, with manifest and metadata for reproducibility and integrity.

    Attributes:
        benchmark_artifact_id: The artifact ID of the associated benchmark.
        model: (Optional) The Model artifact used to generate these predictions.
        zarr_root_path: Path to the Zarr archive containing the predictions.
        column_types: (Optional) A dictionary mapping column names to expected types.
    """

    _artifact_type = "prediction"

    benchmark_artifact_id: str
    model: Model | None = None
    zarr_root_path: str
    _zarr_manifest_path: str | None = None
    _zarr_manifest_md5sum: str | None = None
    column_types: dict[str, type] = {}

    # =====================
    # Validators
    # =====================
    @model_validator(mode="after")
    def _validate_predictions_zarr_structure(self):
        """
        Ensures the Zarr archive for predictions is well-formed:
        - All arrays/groups at the root have the same length.
        - Each group has an __index__ array, its length matches the group size (excluding the index), and all keys in the index exist in the group.
        """
        # Check group index arrays
        for group in self.zarr_root.group_keys():
            if _INDEX_ARRAY_KEY not in self.zarr_root[group].array_keys():
                raise ValueError(f"Group {group} does not have an index array (__index__).")
            index_arr = self.zarr_root[group][_INDEX_ARRAY_KEY]
            if len(index_arr) != len(self.zarr_root[group]) - 1:
                raise ValueError(
                    f"Length of index array for group {group} does not match the size of the group (excluding index)."
                )
            if any(x not in self.zarr_root[group] for x in index_arr):
                raise ValueError(f"Keys of index array for group {group} do not match the group members.")
        # Check all arrays/groups at root have the same length
        lengths = {len(self.zarr_root[k]) for k in self.zarr_root.array_keys()}
        lengths.update({len(self.zarr_root[k][_INDEX_ARRAY_KEY]) for k in self.zarr_root.group_keys()})
        if len(lengths) > 1:
            raise ValueError(
                f"All arrays or groups in the root should have the same length, found the following lengths: {lengths}"
            )
        return self

    # =====================
    # Properties
    # =====================
    @property
    def columns(self):
        return list(self.zarr_root.keys())

    @property
    def dtypes(self):
        dtypes = {}
        for arr in self.zarr_root.array_keys():
            dtypes[arr] = self.zarr_root[arr].dtype
        for group in self.zarr_root.group_keys():
            dtypes[group] = object
        return dtypes

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
            zarr_manifest_path = generate_zarr_manifest(
                self.zarr_root_path, getattr(self, "_cache_dir", None)
            )
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

    # =====================
    # Main API Methods
    # =====================
    def set_prediction_value(self, column: str, row: int, value: Any):
        """
        Set a prediction value for a given column and row in the Zarr archive.
        Validates the column exists and, if column_types is set, checks the value type.
        """
        if column not in self.columns:
            raise KeyError(f"Column '{column}' not defined in predictions.")
        expected_type = self.column_types.get(column)
        if expected_type and not isinstance(value, expected_type):
            raise TypeError(f"Value for column '{column}' must be of type {expected_type}, got {type(value)}")
        self.zarr_root[column][row] = value

    def get_prediction_value(self, column: str, row: int):
        """
        Get a prediction value for a given column and row from the Zarr archive.
        """
        if column not in self.columns:
            raise KeyError(f"Column '{column}' not defined in predictions.")
        return self.zarr_root[column][row]

    @classmethod
    def create_zarr_from_dict(
        cls, path: str, data: dict[str, Any], column_types: dict[str, type] | None = None, **zarr_kwargs
    ):
        """
        Create a Zarr archive at the given path from a dict mapping column names to arrays or lists.
        Uses dtype=object for columns with complex types.
        Returns the path to the created Zarr archive.
        """
        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store)
        column_types = column_types or {}
        for col, arr in data.items():
            dtype = (
                object
                if column_types.get(col) not in (float, int, str, bool, bytes, None)
                else getattr(arr, "dtype", None) or type(arr[0])
            )
            root.create_dataset(col, data=arr, dtype=dtype, **zarr_kwargs)
        zarr.consolidate_metadata(store)
        return path

    def upload_to_hub(
        self,
        owner: HubOwner | str | None = None,
        parent_artifact_id: str | None = None,
        model: Model | None = None,
    ):
        """
        Uploads the predictions artifact to the Polaris Hub.
        Optionally sets or overrides the model before upload.
        """
        if model is not None:
            self.model = model
        with PolarisHubClient() as client:
            client.upload_prediction(self, owner=owner, parent_artifact_id=parent_artifact_id)

    # =====================
    # Representation
    # =====================
    def __repr__(self):
        return self.model_dump_json(by_alias=True, indent=2)

    def __str__(self):
        return self.__repr__()
