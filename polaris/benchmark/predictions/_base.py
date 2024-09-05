import numpy as np
from polaris.utils.types import IncomingPredictionsType, ListOrArrayType, PredictionsType
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
    field_serializer,
)


class BenchmarkPredictions(BaseModel):
    predictions: PredictionsType
    target_cols: list[str]

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

        if vals is None:
            raise ValueError("predictions must be provided")

        if target_cols is None:
            raise ValueError("target_cols must be provided")

        predictions_in_correct_shape = cls._normalize_predictions(vals, target_cols)
        predictions_with_correct_types = cls._convert_lists_to_numpy_arrays(predictions_in_correct_shape)
        cls._check_column_names(predictions_with_correct_types, target_cols)
        data["predictions"] = predictions_with_correct_types

        return data

    @classmethod
    def _normalize_predictions(
        cls, vals: IncomingPredictionsType, target_cols: list[str]
    ) -> dict[str, dict[str, ListOrArrayType]]:
        """Normalizes the predictions so they are always in the standard format,
        {"test-set-name": {"target-name": np.ndarray}} regardless of how they are provided."""

        if isinstance(vals, (list, np.ndarray)):
            return {"test": cls._set_col_name(vals, target_cols)}
        elif cls._is_multi_task_single_test_set(vals, target_cols):
            return {"test": vals}
        elif cls._is_single_task_multi_test_set(vals, target_cols):
            return {test_set_name: cls._set_col_name(v, target_cols) for test_set_name, v in vals.items()}
        else:
            cls._give_feedback(vals)
        return vals

    @classmethod
    def _convert_lists_to_numpy_arrays(
        cls, predictions: dict[str, dict[str, ListOrArrayType]]
    ) -> PredictionsType:
        """Converts all values in the predictions dictionary to numpy arrays."""

        def convert_to_numpy(v):
            if isinstance(v, (list, np.ndarray)):
                return np.array(v)
            elif isinstance(v, dict):
                return {k: convert_to_numpy(v) for k, v in v.items()}

        return convert_to_numpy(predictions)

    @classmethod
    def _is_multi_task_single_test_set(cls, vals, target_cols) -> bool:
        """Check if the given values are for a multiple-task benchmark with a single
        test set. This is inferred by comparing the target names with the keys of the
        given data. If all keys in the given data match the target column names, we
        assume they are target names (as opposed to test set names for a single-task,
        multiple test set benchmark)."""
        return isinstance(vals, dict) and all(k in target_cols for k in vals.keys())

    @classmethod
    def _is_single_task_multi_test_set(cls, vals, target_cols) -> bool:
        """Check if the given values are for a single-task benchmark with multiple
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
                if not (
                    isinstance(predictions, np.ndarray)
                    or (
                        isinstance(predictions, list)
                        and all(isinstance(x, (int, float)) for x in predictions)
                    )
                ):
                    raise ValueError(
                        f"Invalid predictions for test set '{test_set}', target '{target}'. "
                        "Expected a numpy array or list of numbers."
                    )

    @classmethod
    def _check_column_names(cls, predictions, target_cols):
        """Checks that all target names in the predictions match the given target
        column names."""
        for test_set, targets in predictions.items():
            for target, predictions in targets.items():
                if target not in target_cols:
                    raise ValueError(
                        f"Invalid predictions for test set '{test_set}'. Target '{target}' "
                        f"is not in target_cols: {target_cols}."
                    )
