from typing import Callable
from sklearn.metrics import (
    mean_absolute_error as sklearn_mae,
    mean_squared_error as sklearn_mse,
    accuracy_score as sklearn_acc,
)

METRICS_REGISTRY = {}


class Metric:
    def __init__(self, name: str, fn: Callable, is_multitask: bool = False):
        """
        A Metric within the Polaris ecosystem. Each metric is uniquely identified by its name and per name
        there can only be one object (a singleton instance).

        Args:
            name: A human-readable name for the metric
            fn: The callable to actually compute the metric
            is_multitask: Whether the metric expects a single set of predictions or a dict of predictions.
        """

        if name in METRICS_REGISTRY:
            raise ValueError(
                f"{name} is already registered. To reuse the same metric, use Metric.get_by_name()."
            )
        METRICS_REGISTRY[name] = self

        self.name = name
        self.fn = fn

        self.is_multitask = is_multitask

    @classmethod
    def get_by_name(cls, name: str):
        """Singleton access to an already registered metric"""
        return METRICS_REGISTRY[name]

    @staticmethod
    def list_supported_metrics():
        return sorted(list(METRICS_REGISTRY.keys()))

    def score(self, y_true, y_pred):
        """Compute the metric"""
        return self.fn(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        """For convenience, make metrics callable"""
        return self.score(y_true, y_pred)


# TODO (cwognum):
#  - Add support for more metrics
#  - Any preprocessing needed? For example changing the shape / dtype? Converting from torch tensors or lists?
mean_absolute_error = Metric("mean_absolute_error", sklearn_mae)
mean_squared_error = Metric("mean_squared_error", sklearn_mse)
accuracy = Metric("accuracy", sklearn_acc)
