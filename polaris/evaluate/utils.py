import pandas as pd
import numpy as np

from polaris.utils.context import tmp_attribute_change
from polaris.evaluate import BenchmarkResults, ResultsType

def evaluate_benchmark(model, y_pred, test):
    if not isinstance(test, dict):
        test = {"test": test}

    y_true = {}
    for k, test_subset in test.items():
        with tmp_attribute_change(test_subset, "_hide_targets", False):
            y_true[k] = test_subset.targets

    if not isinstance(y_pred, dict) or all(k in model.target_cols for k in y_pred):
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
        for metric in model.metrics:
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
                    model.target_cols[0],
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

    return BenchmarkResults(results=scores, benchmark_name=model.name, benchmark_owner=model.owner)
