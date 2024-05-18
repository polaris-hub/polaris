import numpy as np
import pandas as pd
from typing import Union

from polaris.dataset import Subset
from polaris.evaluate import BenchmarkResults, ResultsType
from polaris.utils.types import PredictionsType
from polaris.evaluate import Metric

def evaluate_benchmark(y_pred: PredictionsType,
                       test_vals: Subset,
                       target_cols: list[str],
                       benchmark_name: str,
                       benchmark_owner: str,
                       metrics: Union[str, Metric, list[Union[str, Metric]]]):
    if not isinstance(test_vals, dict):
        test = {"test": test_vals}
    else:
        test = test_vals

    y_true = {k: test_subset.targets for k, test_subset in test.items()}

    if not isinstance(y_pred, dict) or all(k in target_cols for k in y_pred):
        y_pred = {"test": y_pred}

    if any(k not in y_pred for k in test.keys()):
        raise KeyError(
            f"Missing keys for at least one of the test sets. Expecting: {sorted(test.keys())}"
        )

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
                scores.loc[len(scores)] = (
                    test_label,
                    target_cols[0],
                    metric,
                    score,
                )
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

    return BenchmarkResults(results=scores,
                            benchmark_name=benchmark_name,
                            benchmark_owner=benchmark_owner)
