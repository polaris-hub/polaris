import numpy as np
import pandas as pd
from numpy.typing import NDArray

from polaris.evaluate import BenchmarkPredictions, BenchmarkResults, Metric
from polaris.utils.types import IncomingPredictionsType


def _optionally_get(preds: BenchmarkPredictions | None, keys: list[str] | str) -> dict | None:
    """
    Returns the value in a nested dictionary associated with a sequence of keys
    if it exists, otherwise return None
    """
    if preds is None:
        return None

    if not isinstance(keys, list):
        keys = [keys]

    d = preds.predictions
    for k in keys:
        d = d.get(k)
        if d is None:
            return None
    return d


def _safe_mask(
    preds: BenchmarkPredictions | None,
    mask: NDArray[np.bool_],
    keys: list[str],
) -> NDArray[np.float64] | None:
    """
    Mask a prediction array if it exists in a nested array. Otherwise return None
    """
    v = _optionally_get(preds, keys)
    if v is None:
        return None
    return v[mask]


def mask_index(input_values):
    if np.issubdtype(input_values.dtype, np.number):
        mask = ~np.isnan(input_values)
    else:
        # Create a mask to identify NaNs
        mask = np.full(input_values.shape, True, dtype=bool)
        # Iterate over the array to identify NaNs
        for index, value in np.ndenumerate(input_values):
            # Convert to float and check if it's NaN
            if value is None:
                mask[index] = False
    return mask


def evaluate_benchmark(
    target_cols: list[str],
    test_set_labels: list[str],
    test_set_sizes: dict[str, int],
    metrics: list[Metric],
    y_true: IncomingPredictionsType,
    y_pred: IncomingPredictionsType | None = None,
    y_prob: IncomingPredictionsType | None = None,
):
    """
    Utility function that contains the evaluation logic for a benchmark
    """

    # Normalize the ground truth and predictions to a consistent, internal representation.
    # Format is a two-level dictionary: {test_set_label: {target_label: np.ndarray}}
    y_true = BenchmarkPredictions(
        predictions=y_true,
        target_labels=target_cols,
        test_set_labels=test_set_labels,
        test_set_sizes=test_set_sizes,
    )
    if y_pred is not None:
        y_pred = BenchmarkPredictions(
            predictions=y_pred,
            target_labels=target_cols,
            test_set_labels=test_set_labels,
            test_set_sizes=test_set_sizes,
        )
    if y_prob is not None:
        y_prob = BenchmarkPredictions(
            predictions=y_prob,
            target_labels=target_cols,
            test_set_labels=test_set_labels,
            test_set_sizes=test_set_sizes,
        )

    # Compute the results
    # Results are saved in a tabular format. For more info, see the BenchmarkResults docs.
    scores = pd.DataFrame(columns=BenchmarkResults.RESULTS_COLUMNS)

    # For every test set...
    for test_label, y_true_test in y_true.predictions.items():
        # For every metric...
        for metric in metrics:
            if metric.is_multitask:
                # Multi-task but with a metric across targets
                score = metric(
                    y_true=y_true_test,
                    y_pred=_optionally_get(y_pred, test_label),
                    y_prob=_optionally_get(y_prob, test_label),
                )

                scores.loc[len(scores)] = (test_label, "aggregated", metric, score)
                continue

            # Otherwise, for every target...
            for target_label, y_true_target in y_true_test.items():
                # Single-task metrics for a multi-task benchmark
                # In such a setting, there can be NaN values, which we thus have to filter out.

                mask = mask_index(y_true_target)

                score = metric(
                    y_true=y_true_target[mask],
                    y_pred=_safe_mask(y_pred, mask, [test_label, target_label]),
                    y_prob=_safe_mask(y_prob, mask, [test_label, target_label]),
                )

                scores.loc[len(scores)] = (test_label, target_label, metric, score)

    return scores
