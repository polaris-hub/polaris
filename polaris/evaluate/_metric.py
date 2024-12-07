from enum import Enum
from typing import Annotated, Callable, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter, field_serializer, field_validator, model_validator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from typing_extensions import Self

from polaris.dataset._subset import Subset
from polaris.evaluate._predictions import BenchmarkPredictions
from polaris.evaluate.metrics import (
    absolute_average_fold_error,
    average_precision_score,
    cohen_kappa_score,
    pearsonr,
    spearman,
)
from polaris.evaluate.metrics.docking_metrics import rmsd_coverage
from polaris.utils.types import DirectionType, PredictionKwargs

GroundTruth: TypeAlias = Subset


def prepare_predictions(
    y_pred: BenchmarkPredictions,
    y_prob: BenchmarkPredictions,
    y_type: PredictionKwargs,
) -> BenchmarkPredictions:
    """Check that the correct type of predictions are passed to the metric."""

    if y_pred is None and y_prob is None:
        raise ValueError("Neither `y_pred` nor `y_prob` is specified.")

    if y_type == "y_pred":
        if y_pred is None:
            raise ValueError("Metric requires `y_pred` input.")
        pred = y_pred

    else:
        if y_prob is None:
            raise ValueError("Metric requires `y_prob` input.")
        pred = y_prob

    return pred


def prepare_metric_kwargs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_type: PredictionKwargs,
) -> BenchmarkPredictions:
    mask = mask_index(y_true)
    return {"y_true": y_true[mask], y_type: y_pred[mask]}


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


class MetricInfo(BaseModel):
    """
    Metric metadata

    Attributes:
        fn: The callable that actually computes the metric.
        is_multitask: Whether the metric expects a single set of predictions or a dict of predictions.
        kwargs: Additional parameters required for the metric.
        direction: The direction for ranking of the metric,  "max" for maximization and "min" for minimization.
        y_type: The type of predictions expected by the metric interface.
    """

    fn: Callable
    is_multitask: bool = False
    kwargs: dict = Field(default_factory=dict)
    direction: DirectionType
    y_type: PredictionKwargs = "y_pred"

    @field_validator("is_multitask")
    @classmethod
    def disable_multitask_metrics(cls, value):
        raise NotImplementedError(
            "Multitask metrics are not yet supported and will require non-obvious changes to the Metric interface."
        )


class BaseMetric(Enum):
    """
    A metric within the Polaris ecosystem is uniquely identified by its name
    and is associated with additional metadata in a `MetricInfo` instance.

    Implemented as an enum.
    """

    # TODO (cwognum):
    #  - Any preprocessing needed? For example changing the shape / dtype? Converting from torch tensors or lists?

    # regression
    mean_absolute_error = MetricInfo(fn=mean_absolute_error, direction="min")
    mean_squared_error = MetricInfo(fn=mean_squared_error, direction="min")
    r2 = MetricInfo(fn=r2_score, direction="max")
    pearsonr = MetricInfo(fn=pearsonr, direction="max")
    spearmanr = MetricInfo(fn=spearman, direction="max")
    explained_var = MetricInfo(fn=explained_variance_score, direction="max")
    absolute_average_fold_error = MetricInfo(fn=absolute_average_fold_error, direction=1)

    # binary and multiclass classification
    accuracy = MetricInfo(fn=accuracy_score, direction="max")
    balanced_accuracy = MetricInfo(fn=balanced_accuracy_score, direction="max")
    mcc = MetricInfo(fn=matthews_corrcoef, direction="max")
    cohen_kappa = MetricInfo(fn=cohen_kappa_score, direction="max")
    pr_auc = MetricInfo(fn=average_precision_score, direction="max", y_type="y_score")

    # binary only
    f1 = MetricInfo(fn=f1_score, kwargs={"average": "binary"}, direction="max")
    roc_auc = MetricInfo(fn=roc_auc_score, direction="max", y_type="y_score")

    # multiclass tasks only
    f1_macro = MetricInfo(fn=f1_score, kwargs={"average": "macro"}, direction="max")
    f1_micro = MetricInfo(fn=f1_score, kwargs={"average": "micro"}, direction="max")
    roc_auc_ovr = MetricInfo(
        fn=roc_auc_score, kwargs={"multi_class": "ovr"}, direction="max", y_type="y_score"
    )
    roc_auc_ovo = MetricInfo(
        fn=roc_auc_score, kwargs={"multi_class": "ovo"}, direction="max", y_type="y_score"
    )
    # TODO: add metrics to handle multitask multiclass predictions.

    # docking related metrics
    rmsd_coverage = MetricInfo(fn=rmsd_coverage, direction="max", y_type="y_pred")


class Metric(BaseModel):
    metric: BaseMetric
    kind: Literal[None] = None
    name: str | None = None

    @field_validator("metric", mode="before")
    @classmethod
    def validate_metric(cls, value) -> BaseMetric:
        if not isinstance(value, BaseMetric):
            value = BaseMetric[value]
        return value

    @field_serializer("metric")
    def serialize_metric(self, value: BaseMetric) -> str:
        return value.name

    @model_validator(mode="after")
    def set_name_if_none(self) -> Self:
        if self.name is None:
            kind = self.kind or ""
            self.name = f"{kind}_{self.metric.name}"
        return self

    @property
    def fn(self) -> Callable:
        """The callable that actually computes the metric"""
        return self.metric.value.fn

    @property
    def is_multitask(self) -> bool:
        """Whether the metric expects a single set of predictions or a dict of predictions."""
        return self.metric.value.is_multitask

    @property
    def y_type(self) -> bool:
        """Whether the metric expects preditive probablities."""
        return self.metric.value.y_type

    def score(
        self,
        y_true: GroundTruth,
        y_pred: BenchmarkPredictions | None = None,
        y_prob: BenchmarkPredictions | None = None,
    ) -> float:
        """Endpoint for computing the metric.

        For convenience, calling a `Metric` will result in this method being called.

        ```python
        metric = Metric(metric="mean_absolute_error")
        assert metric.score(y_true=first, y_pred=second) == metric(y_true=first, y_pred=second)
        ```
        """
        pred = prepare_predictions(y_pred, y_prob, self.y_type)
        kwargs = prepare_metric_kwargs(
            y_true.as_array("y"),
            pred.flatten(),
            self.y_type,
        )
        return self.fn(**kwargs, **self.metric.value.kwargs)

    def __call__(
        self,
        y_true: GroundTruth,
        y_pred: BenchmarkPredictions | None = None,
        y_prob: BenchmarkPredictions | None = None,
    ) -> float:
        return self.score(y_true, y_pred, y_prob)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        match other:
            case Metric():
                return other.name == self.name
            case _:
                return False


class GroupedMetricConfig(BaseModel):
    group_by: str
    on_error: Literal["ignore", "raise", "default"] = "raise"
    default: float | None = None
    aggregation: Literal["mean", "median"] = "mean"


class GroupedMetric(Metric):
    """
    A GroupedMetric is a Metric that evaluates predictions grouped by a specific column.

    Attributes:
        metric: The callable metric to evaluate.
        config: Configuration for the grouped metric.
    """

    kind: Literal["grouped"] = "grouped"
    config: GroupedMetricConfig

    @model_validator(mode="after")
    def check_default_required(self) -> Self:
        handling = self.config.on_error
        default = self.config.default
        if handling == "default" and default is None:
            raise ValueError("Default value must be provided when on_error is 'default'")
        return self

    def score(
        self,
        y_true: GroundTruth,
        y_pred: BenchmarkPredictions | None = None,
        y_prob: BenchmarkPredictions | None = None,
    ) -> float:
        pred = prepare_predictions(y_pred, y_prob, self.y_type)

        df = y_true.as_dataframe()
        df[self.y_type] = pred.flatten()

        scores = []
        for _, group in df.groupby(self.config.group_by):
            y_true_group = group[y_true.target_cols[0]].values
            y_pred_group = group[self.y_type].values

            kwargs = prepare_metric_kwargs(
                y_true_group,
                y_pred_group,
                self.y_type,
            )

            try:
                score = self.fn(**kwargs, **self.metric.value.kwargs)
            except Exception:
                if self.config.on_error == "ignore":
                    continue
                elif self.config.on_error == "raise":
                    raise
                score = self.config.default
            scores.append(score)

        if self.config.aggregation == "mean":
            score = np.mean(scores)
        else:
            score = np.median(scores)
        return score


MetricType: TypeAlias = Metric | GroupedMetric
AnnotatedMetricType = Annotated[MetricType, Field(..., discriminator="kind")]


def instantiate_metric(m: MetricType | str | dict):
    if isinstance(m, str):
        return Metric(metric=m)

    if isinstance(m, dict):
        if "kind" not in m:
            m["kind"] = None
        adapter = TypeAdapter(AnnotatedMetricType)
        return adapter.validate_python(m)

    return m
