from typing import Callable
from sklearn.metrics import (
    mean_absolute_error as sklearn_mae,
    accuracy_score as sklearn_acc,
)

_METRICS_REGISTRY = {}


class Metric:
    def __init__(self, name: str, fn: Callable):
        if name in _METRICS_REGISTRY:
            raise ValueError(f"Metric {name} already registered.")
        _METRICS_REGISTRY[name] = self

        self.name = name
        self.fn = fn

    @classmethod
    def get_by_name(cls, name: str):
        return _METRICS_REGISTRY[name]

    def score(self, y_true, y_pred):
        return self.fn(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        return self.score(y_true, y_pred)


# TODO (cwognum):
#  - Add support for more metrics
#  - Can we come up with a more systematic, robust way of specifying default metrics?
mean_absolute_error = Metric("mean_absolute_error", sklearn_mae)
accuracy = Metric("accuracy", sklearn_acc)
