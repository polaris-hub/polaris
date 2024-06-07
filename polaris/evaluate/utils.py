import numpy as np
import pandas as pd
from typing import Union

from polaris.evaluate import BenchmarkResults, ResultsType
from polaris.utils.types import PredictionsType
from polaris.evaluate import Metric


def is_multi_task_single_test_set(vals: PredictionsType, target_cols: list[str]):
    """Check if the given values are for a multiple-task benchmark with a single
    test set. This is inferred by comparing the target names with the keys of the
    given data. If all keys in the given data match the target column names, we
    assume they are target names (as opposed to test set names for a single-task,
    multiple test set benchmark)."""
    return not isinstance(vals, dict) or set(vals.keys()) == set(target_cols)


def serialize_predictions(
    predictions: PredictionsType,
):
    """Used to serialize a Predictions object such that it can be sent over the wire during
    external evaluation for competitions"""

    if isinstance(predictions, np.ndarray):
        return predictions.tolist()
    elif isinstance(predictions, list):
        return predictions
    elif isinstance(predictions, dict):
        for key, val in predictions.items():
            predictions[key] = serialize_predictions(val)
        return predictions
    else:
        raise TypeError(f"Unexpected type of {type(predictions)} encountered in Predictions object.")


def deserialize_predictions(
    predictions: PredictionsType,
):
    """Used to deserialize a Predictions object such that it can be converted to the
    expected type of numpy array during external evaluation for competitions"""

    if isinstance(predictions, np.ndarray):
        return predictions
    elif isinstance(predictions, list):
        return np.array(predictions)
    elif isinstance(predictions, dict):
        for key, val in predictions.items():
            predictions[key] = deserialize_predictions(val)
        return predictions
    else:
        raise TypeError(f"Unexpected type of {type(predictions)} encountered in Predictions object.")


def normalize_predictions_type(vals: np.array, target_cols: list[str]):
    if is_multi_task_single_test_set(vals, target_cols):
        if isinstance(vals, dict):
            return {"test": vals}
        else:
            return {"test": {target_cols[0]: np.array(vals)}}
    else:
        return vals


def evaluate_benchmark(
    y_pred: PredictionsType,
    y_true: PredictionsType,
    target_cols: list[str],
    metrics: Union[str, Metric, list[Union[str, Metric]]],
):
    y_true = normalize_predictions_type(y_true, target_cols)
    y_pred = normalize_predictions_type(y_pred, target_cols)

    if set(y_true.keys()) != set(y_pred.keys()):
        raise KeyError(f"Missing keys for at least one of the test sets. Expecting: {sorted(y_true.keys())}")

    # Results are saved in a tabular format. For more info, see the BenchmarkResults docs.
    scores: ResultsType = pd.DataFrame(columns=BenchmarkResults.RESULTS_COLUMNS)

    # For every test set...
    for test_label, y_true_subset in y_true.items():
        # For every metric...
        for metric in metrics:
            if metric.is_multitask:
                # Multi-task but with a metric across targets
                score = metric(y_true=y_true_subset, y_pred=y_pred[test_label])
                scores.loc[len(scores)] = (test_label, "aggregated", metric, score)
                continue

            if not isinstance(y_true_subset, dict):
                # Single task
                score = metric(y_true=y_true_subset, y_pred=y_pred[test_label])
                scores.loc[len(scores)] = (test_label, target_cols[0], metric, score)
                continue

            # Otherwise, for every target...
            for target_label, y_true_target in y_true_subset.items():
                # Single-task metrics for a multi-task benchmark
                # In such a setting, there can be NaN values, which we thus have to filter out.
                mask = ~np.isnan(y_true_target)
                score = metric(
                    y_true=y_true_target[mask],
                    y_pred=y_pred[test_label][target_label][mask],
                )
                scores.loc[len(scores)] = (test_label, target_label, metric, score)

    return scores
