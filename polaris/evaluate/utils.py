import numpy as np
import pandas as pd
from typing import Optional

from polaris.evaluate import BenchmarkResults, ResultsType
from polaris.benchmark.predictions import BenchmarkPredictions
from polaris.utils.types import IncomingPredictionsType, PredictionsType
from polaris.evaluate import Metric
from numpy.typing import NDArray


def safe_mask(input_values: PredictionsType, test_label: str, target_label: str, mask: NDArray[np.bool_]):
    if (
        input_values is None
        or input_values.predictions.get(test_label) is None
        or input_values.predictions[test_label].get(target_label) is None
    ):
        return None
    else:
        return input_values.predictions[test_label][target_label][mask]


def evaluate_benchmark(
    target_cols: list[str],
    test_set_names: list[str],
    metrics: list[Metric],
    y_true: IncomingPredictionsType,
    y_pred: Optional[IncomingPredictionsType] = None,
    y_prob: Optional[IncomingPredictionsType] = None,
):
    y_true = BenchmarkPredictions(predictions=y_true, target_cols=target_cols, test_set_names=test_set_names)
    if y_pred is not None:
        y_pred = BenchmarkPredictions(
            predictions=y_pred, target_cols=target_cols, test_set_names=test_set_names
        )
    if y_prob is not None:
        y_prob = BenchmarkPredictions(
            predictions=y_prob, target_cols=target_cols, test_set_names=test_set_names
        )

    if y_pred and set(y_true.predictions.keys()) != set(y_pred.predictions.keys()):
        raise KeyError(
            f"Missing keys for at least one of the test sets. Expecting: {sorted(y_true.predictions.keys())}"
        )

    # Results are saved in a tabular format. For more info, see the BenchmarkResults docs.
    scores: ResultsType = pd.DataFrame(columns=BenchmarkResults.RESULTS_COLUMNS)

    # For every test set...
    for test_label, y_true_test in y_true.predictions.items():
        # For every metric...
        for metric in metrics:
            if metric.is_multitask:
                # Multi-task but with a metric across targets
                score = metric(
                    y_true=y_true_test,
                    y_pred=y_pred.predictions.get(test_label) if y_pred is not None else None,
                    y_prob=y_prob.predictions.get(test_label) if y_prob is not None else None,
                )

                scores.loc[len(scores)] = (test_label, "aggregated", metric, score)
                continue

            if not isinstance(y_true_test, dict):
                # Single task
                score = metric(
                    y_true=y_true_test,
                    y_pred=y_pred.predictions.get(test_label) if y_pred is not None else None,
                    y_prob=y_prob.predictions.get(test_label) if y_prob is not None else None,
                )
                scores.loc[len(scores)] = (test_label, target_cols[0], metric, score)
                continue

            # Otherwise, for every target...
            for target_label, y_true_target in y_true_test.items():
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
