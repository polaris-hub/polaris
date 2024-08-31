import numpy as np
import pandas as pd
from typing import Optional

from polaris.evaluate import BenchmarkResults, ResultsType
from polaris.benchmark.predictions import BenchmarkPredictions
from polaris.utils.types import PredictionsType
from polaris.evaluate import Metric
from numpy.typing import NDArray


def safe_mask(
    input_values: dict | dict[str, dict], test_label: str, target_label: str, mask: NDArray[np.bool_]
):
    if (
        input_values is None
        or input_values.get(test_label) is None
        or input_values[test_label].get(target_label) is None
    ):
        return None
    else:
        return input_values[test_label][target_label][mask]


def evaluate_benchmark(
    target_cols: list[str],
    metrics: list[Metric],
    y_true: PredictionsType,
    y_pred: Optional[PredictionsType] = None,
    y_prob: Optional[PredictionsType] = None,
):
    y_true = BenchmarkPredictions(predictions=y_true, target_cols=target_cols)
    y_pred = BenchmarkPredictions(predictions=y_pred, target_cols=target_cols)
    y_prob = BenchmarkPredictions(predictions=y_prob, target_cols=target_cols)

    if y_pred and set(y_true.keys()) != set(y_pred.keys()):
        raise KeyError(f"Missing keys for at least one of the test sets. Expecting: {sorted(y_true.keys())}")

    # Results are saved in a tabular format. For more info, see the BenchmarkResults docs.
    scores: ResultsType = pd.DataFrame(columns=BenchmarkResults.RESULTS_COLUMNS)

    # For every test set...
    for test_label, y_true_subset in y_true.items():
        # For every metric...
        for metric in metrics:
            if metric.is_multitask:
                # Multi-task but with a metric across targets
                score = metric(
                    y_true=y_true_subset, y_pred=y_pred.get(test_label), y_prob=y_prob.get(test_label)
                )

                scores.loc[len(scores)] = (test_label, "aggregated", metric, score)
                continue

            if not isinstance(y_true_subset, dict):
                # Single task
                score = metric(
                    y_true=y_true_subset, y_pred=y_pred.get(test_label), y_prob=y_prob.get(test_label)
                )
                scores.loc[len(scores)] = (test_label, target_cols[0], metric, score)
                continue

            # Otherwise, for every target...
            for target_label, y_true_target in y_true_subset.items():
                # Single-task metrics for a multi-task benchmark
                # In such a setting, there can be NaN values, which we thus have to filter out.
                mask = ~np.isnan(y_true_target)
                score = metric(
                    y_true=y_true_target[mask],
                    y_pred=safe_mask(y_pred, test_label, target_label, mask),
                    y_prob=safe_mask(y_prob, test_label, target_label, mask),
                )

                scores.loc[len(scores)] = (test_label, target_label, metric, score)

    return scores
