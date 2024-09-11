import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    model_validator,
)

from polaris.utils.types import IncomingPredictionsType, ListOrArrayType, PredictionsType


class BenchmarkPredictions(BaseModel):
    """
    Base model to represent predictions in the Polaris code base.

    Guided by [Postel's Law](https://en.wikipedia.org/wiki/Robustness_principle),
    this class normalized different formats to a single, internal representation.

    Attributes:
        predictions: The predictions for the benchmark.
        target_cols: The target columns for the associated benchmark.
        test_set_names: The names of the test sets for the associated benchmark.

    """

    predictions: PredictionsType
    target_cols: list[str]
    test_set_names: list[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("predictions")
    def _serialize_predictions(self, predictions: PredictionsType):
        """Converts all numpy values in the predictions dictionary to lists so
        they can be sent over the wire."""

        def convert_to_list(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, dict):
                return {k: convert_to_list(v) for k, v in v.items()}

        return convert_to_list(predictions)

    @model_validator(mode="before")
    @classmethod
    def validate_predictions(cls, data):
        vals = data.get("predictions")
        target_cols = data.get("target_cols")
        test_set_names = data.get("test_set_names")

        if vals is None:
            raise ValueError("predictions must be provided")

        if target_cols is None:
            raise ValueError("target_cols must be provided")

        predictions_in_correct_shape = cls._normalize_predictions(vals, target_cols, test_set_names)
        predictions_with_correct_types = cls._convert_number_lists_to_numpy_arrays(
            predictions_in_correct_shape
        )
        cls._check_column_names(predictions_with_correct_types, target_cols)
        data["predictions"] = predictions_with_correct_types

        return data

    @classmethod
    def _normalize_predictions(
        cls, vals: IncomingPredictionsType, target_cols: list[str], test_set_names: list[str]
    ) -> dict[str, dict[str, ListOrArrayType]]:
        """
        Normalizes the predictions so they are always in the standard format,
        {"test-set-name": {"target-name": np.ndarray}} regardless of how they are provided.
        """

        if isinstance(vals, (list, np.ndarray)):
            if len(test_set_names) != 1:
                raise ValueError(
                    f"Missing predictions. Single array/list of predictions provided "
                    f"but multiple test sets expected with names {test_set_names}."
                )
            if len(target_cols) != 1:
                raise ValueError(
                    f"Missing targets. Single array/list of predictions provided "
                    f"but multiple targets expected with names {target_cols}."
                )
            return {"test": cls._set_col_name(vals, target_cols)}

        elif cls._is_multi_task_single_test_set(vals, target_cols):
            if len(test_set_names) != 1:
                raise ValueError(
                    f"Missing test sets. Single dictionary of predictions provided "
                    f"but multiple test sets expected with names {test_set_names}."
                )
            return {"test": vals}

        elif cls._is_single_task_multi_test_set(vals, target_cols):
            return cls._validate_all_test_sets(vals, target_cols, test_set_names)

        else:
            cls._give_feedback(vals)
        return vals

    @classmethod
    def _validate_all_test_sets(cls, vals: dict, target_cols: list[str], test_set_labels: list[str]):
        if set(vals.keys()) != set(test_set_labels):
            raise ValueError(f"Predictions must be provided for all test sets: {test_set_labels}")
        return {test_set: cls._set_col_name(v, target_cols) for test_set, v in vals.items()}

    @classmethod
    def _convert_number_lists_to_numpy_arrays(
        cls, predictions: dict[str, dict[str, ListOrArrayType]]
    ) -> PredictionsType:
        """Converts all numeric values in the predictions to numpy arrays."""

        def convert_numbers_to_numpy(v):
            if isinstance(v, list) and all(isinstance(item, (int, float)) for item in v):
                return np.array(v)
            elif isinstance(v, np.ndarray) or isinstance(v, list):
                return v
            elif isinstance(v, dict):
                return {k: convert_numbers_to_numpy(v) for k, v in v.items()}

        return convert_numbers_to_numpy(predictions)

    @classmethod
    def _is_multi_task_single_test_set(cls, vals, target_cols) -> bool:
        """
        Check if the given values are for a multiple-task benchmark with a single
        test set. This is inferred by comparing the target names with the keys of the
        given data. If all keys in the given data match the target column names, we
        assume they are target names (as opposed to test set names for a single-task,
        multiple test set benchmark).
        """
        return isinstance(vals, dict) and all(k in target_cols for k in vals.keys())

    @classmethod
    def _is_single_task_multi_test_set(cls, vals, target_cols) -> bool:
        """
        Check if the given values are for a single-task benchmark with multiple
        test sets. This is inferred by comparing the target names with the keys of the
        given data. If there is a single target column but more than one dictionary entry,
        and none of the dictionary keys match the target column name, we assume the
        dictionary keys are test set names and the values are the predictions for each test set.
        """
        return (
            len(target_cols) == 1
            and isinstance(vals, dict)
            and len(vals.keys()) > 1
            and all(k != target_cols[0] for k in vals.keys())
        )

    @classmethod
    def _set_col_name(cls, vals, target_cols: list[str]):
        if isinstance(vals, list):
            vals = np.array(vals)
        if isinstance(vals, dict):
            return vals
        else:
            return {target_cols[0]: vals}

    @classmethod
    def _give_feedback(cls, vals):
        for test_set, targets in vals.items():
            if not isinstance(targets, dict):
                raise ValueError(
                    f"Invalid structure for test set '{test_set}'. "
                    "Expected a dictionary of {col_name: predictions}"
                )
            for target, predictions in targets.items():
                if not cls._valid_incoming_predictions(predictions):
                    raise ValueError(
                        f"Invalid predictions for test set '{test_set}', target '{target}'. "
                        "Expected a numpy array or list."
                    )

    @classmethod
    def _valid_incoming_predictions(cls, predictions: IncomingPredictionsType) -> bool:
        return isinstance(predictions, np.ndarray) or isinstance(predictions, list)

    @classmethod
    def _check_column_names(cls, predictions, target_cols):
        """Checks that all target names in the predictions match the given target
        column names."""
        for test_set, targets in predictions.items():
            for target in targets.keys():
                if target not in target_cols:
                    raise ValueError(
                        f"Invalid predictions for test set '{test_set}'. Target '{target}' "
                        f"is not in target_cols: {target_cols}."
                    )
