import numpy as np
import pandas as pd
from numpy.typing import NDArray

from polaris.evaluate import BenchmarkPredictions, BenchmarkResults, Metric, ResultsType
from polaris.evaluate._ground_truth import GroundTruth
from polaris.utils.types import IncomingPredictionsType


def _optionally_subset(
    preds: BenchmarkPredictions | None,
    test_set_labels: list[str] | str,
    target_labels: list[str] | str,
) -> dict | None:
    """
    Returns the value in a nested dictionary associated with a sequence of keys
    if it exists, otherwise return None
    """
    if preds is None:
        return None

    if not isinstance(test_set_labels, list):
        test_set_labels = [test_set_labels]

    if not isinstance(target_labels, list):
        target_labels = [target_labels]

    return preds.get_subset(
        test_set_subset=test_set_labels,
        target_subset=target_labels,
    )


def _safe_mask(
    preds: BenchmarkPredictions | None,
    mask: NDArray[np.bool_],
    test_set_labels: list[str] | str,
    target_labels: list[str] | str,
) -> NDArray[np.float64] | None:
    """
    Mask a prediction array if it exists in a nested array. Otherwise return None
    """
    v = _optionally_subset(
        preds,
        test_set_labels=test_set_labels,
        target_labels=target_labels,
    )

    if v is None:
        return None
    return v.mask(mask)


def mask_index(input_values):
    if np.issubdtype(input_values.dtype, np.number):
        mask = ~np.isnan(input_values)
    else:
        # Create a mask to identify NaNs
        mask = np.full(input_values.shape, True, dtype=bool)
        # Iterate over the array to identify NaNs
        for index, value in np.ndenumerate(input_values):
            # Any None value is NaN
            if value is None:
                mask[index] = False
    return mask


def evaluate_benchmark(
    target_cols: list[str],
    test_set_labels: list[str],
    test_set_sizes: dict[str, int],
    metrics: list[Metric],
    y_true: dict[str, GroundTruth],
    y_pred: IncomingPredictionsType | None = None,
    y_prob: IncomingPredictionsType | None = None,
):
    """
    Utility function that contains the evaluation logic for a benchmark
    """

    # Normalize the and predictions to a consistent, internal representation.
    # Format is a two-level dictionary: {test_set_label: {target_label: np.ndarray}}
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
    scores: ResultsType = pd.DataFrame(columns=BenchmarkResults.RESULTS_COLUMNS)

    # For every test set...
    for test_label in test_set_labels:
        # For every metric...
        for metric in metrics:
            if metric.is_multitask:
                # Multi-task but with a metric across targets
                score = metric(
                    y_true=y_true[test_label],
                    y_pred=_optionally_subset(y_pred, test_set_labels=test_label),
                    y_prob=_optionally_subset(y_prob, test_set_labels=test_label),
                )

                scores.loc[len(scores)] = (test_label, "aggregated", metric, score)
                continue

            # Otherwise, for every target...
            for target_label in target_cols:
                # Single-task metrics for a multi-task benchmark
                # In such a setting, there can be NaN values, which we thus have to filter out.
                y_true_subset = y_true[test_label]
                y_true_array = y_true_subset.as_array(target_label)
                mask = mask_index(y_true_array)

                y_true_subset.set_mask(mask)

                score = metric(
                    y_true=y_true_subset,
                    y_pred=_safe_mask(y_pred, mask, test_set_labels=test_label, target_labels=target_label),
                    y_prob=_safe_mask(y_prob, mask, test_set_labels=test_label, target_labels=target_label),
                )

                scores.loc[len(scores)] = (test_label, target_label, metric, score)

    return scores
