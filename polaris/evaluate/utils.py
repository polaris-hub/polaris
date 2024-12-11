import pandas as pd

from polaris.dataset._subset import Subset
from polaris.evaluate import BenchmarkPredictions, BenchmarkResults, Metric
from polaris.utils.types import IncomingPredictionsType


def _optionally_subset(
    preds: BenchmarkPredictions | None,
    test_set_labels: list[str] | str,
    target_labels: list[str] | str,
) -> BenchmarkPredictions | None:
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


def evaluate_benchmark(
    target_cols: list[str],
    test_set_labels: list[str],
    test_set_sizes: dict[str, int],
    metrics: set[Metric],
    y_true: dict[str, Subset],
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
    scores = pd.DataFrame(columns=BenchmarkResults.RESULTS_COLUMNS)

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
                score = metric(
                    y_true=y_true[test_label].filter_targets(target_label),
                    y_pred=_optionally_subset(y_pred, test_set_labels=test_label, target_labels=target_label),
                    y_prob=_optionally_subset(y_prob, test_set_labels=test_label, target_labels=target_label),
                )

                scores.loc[len(scores)] = (test_label, target_label, metric.name, score)

    return scores
