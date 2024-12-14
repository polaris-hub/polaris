import json
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)
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
    """
    Check that the correct type of predictions are passed to the metric.

    Args:
        y_pred: The predicted target values, if any.
        y_prob: The predicted target probabilities, if any.
        y_type: The type of predictions expected by the metric interface
    """

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


def mask_index(input_values):
    """
    Mask the NaN values in the input array

    Args:
        input_values: The input array to mask.
    """

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


def prepare_metric_kwargs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_type: PredictionKwargs,
) -> BenchmarkPredictions:
    """
    Prepare the arguments for the metric function.

    Args:
        y_true: The true target values of shape (n_samples,)
        y_pred: The predicted target values of shape (n_samples,).
        y_type: The type of predictions expected by the metric interface.
    """
    mask = mask_index(y_true)
    return {"y_true": y_true[mask], y_type: y_pred[mask]}


MetricKind: TypeAlias = Literal["default", "grouped"]

MetricLabel: TypeAlias = Literal[
    "mean_absolute_error",
    "mean_squared_error",
    "r2",
    "pearsonr",
    "spearmanr",
    "explained_var",
    "absolute_average_fold_error",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "cohen_kappa",
    "pr_auc",
    "f1",
    "roc_auc",
    "f1_macro",
    "f1_micro",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "rmsd_coverage",
]


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

    model_config = ConfigDict(frozen=True)


DEFAULT_METRICS: dict[MetricLabel, MetricInfo] = {
    "mean_absolute_error": MetricInfo(fn=mean_absolute_error, direction="min"),
    "mean_squared_error": MetricInfo(fn=mean_squared_error, direction="min"),
    "r2": MetricInfo(fn=r2_score, direction="max"),
    "pearsonr": MetricInfo(fn=pearsonr, direction="max"),
    "spearmanr": MetricInfo(fn=spearman, direction="max"),
    "explained_var": MetricInfo(fn=explained_variance_score, direction="max"),
    "absolute_average_fold_error": MetricInfo(fn=absolute_average_fold_error, direction=1),
    "accuracy": MetricInfo(fn=accuracy_score, direction="max"),
    "balanced_accuracy": MetricInfo(fn=balanced_accuracy_score, direction="max"),
    "mcc": MetricInfo(fn=matthews_corrcoef, direction="max"),
    "cohen_kappa": MetricInfo(fn=cohen_kappa_score, direction="max"),
    "pr_auc": MetricInfo(fn=average_precision_score, direction="max", y_type="y_score"),
    "f1": MetricInfo(fn=f1_score, kwargs={"average": "binary"}, direction="max"),
    "roc_auc": MetricInfo(fn=roc_auc_score, direction="max", y_type="y_score"),
    "f1_macro": MetricInfo(fn=f1_score, kwargs={"average": "macro"}, direction="max"),
    "f1_micro": MetricInfo(fn=f1_score, kwargs={"average": "micro"}, direction="max"),
    "roc_auc_ovr": MetricInfo(
        fn=roc_auc_score, kwargs={"multi_class": "ovr"}, direction="max", y_type="y_score"
    ),
    "roc_auc_ovo": MetricInfo(
        fn=roc_auc_score, kwargs={"multi_class": "ovo"}, direction="max", y_type="y_score"
    ),
    "rmsd_coverage": MetricInfo(fn=rmsd_coverage, direction="max", y_type="y_pred"),
}


class GroupedMetricConfig(BaseModel):
    """
    The configuration for a GroupedMetric

    Attributes:
        group_by: The column to group by.
        on_error: How to handle errors when computing the metric.
        default: The default value to use when an error occurs.
        aggregation: The aggregation method to use when combining the metric scores.
    """

    _kind: MetricKind = PrivateAttr("grouped")

    group_by: str
    on_error: Literal["ignore", "raise", "default"] = "raise"
    default: float | None = None
    aggregation: Literal["mean", "median"] = "mean"

    @model_validator(mode="after")
    def check_default_required(self) -> Self:
        """Check that a default value is provided when on_error is 'default'."""
        handling = self.on_error
        default = self.default
        if handling == "default" and default is None:
            raise ValueError("Default value must be provided when on_error is 'default'")
        return self

    def score(
        self,
        y_true: GroundTruth,
        y_pred: BenchmarkPredictions | None,
        y_prob: BenchmarkPredictions | None,
        info: MetricInfo,
    ) -> float:
        """Compute the metric.

        Args:
            y_true: The true target values.
            y_pred: The predicted target values, if any.
            y_prob: The predicted target probabilities, if any.
            info: The metadata for the metric.
        """

        pred = prepare_predictions(y_pred, y_prob, info.y_type)

        # NOTE (cwognum): We rely on pandas to do the grouping.
        # This does imply a memory bottleneck.

        df = y_true.extend_inputs(self.group_by).as_dataframe()
        df[info.y_type] = pred.flatten()

        # Compute the metric for each group
        scores = []

        for _, group in df.groupby(self.group_by):
            y_true_group = group[y_true.target_cols[0]].values
            y_pred_group = group[info.y_type].values

            kwargs = prepare_metric_kwargs(
                y_true_group,
                y_pred_group,
                info.y_type,
            )

            # A group can be arbitrarily small (e.g. only 1 item), which also has implications
            # for its target distribution (e.g. only the positive class). These conditions can
            # lead to errors in the metric computation, which we handle here.
            try:
                score = info.fn(**kwargs, **info.kwargs)
            except Exception:
                if self.on_error == "ignore":
                    continue
                elif self.on_error == "raise":
                    raise
                score = self.default
            scores.append(score)

        # Aggregate the per-group score in a single score.
        if self.aggregation == "mean":
            score = np.mean(scores)
        else:
            score = np.median(scores)
        return score


class Metric(BaseModel):
    """
    A Metric in Polaris.

    A metric consists of a default metric, which is a callable labeled with additional metadata, as well as a config.
    The config can change how the metric is computed, for example by grouping the data before computing the metric.

    Attributes:
        label: The actual callable that is at the core of the metric implementation.
        custom_name: A optional, custom name of the metric. Names should be unique within the context of a benchmark.
        config: For more complex metrics, this object should hold all parameters for the metric.
        fn: The callable that actually computes the metric, automatically set based on the label.
        is_multitask: Whether the metric expects a single set of predictions or a dict of predictions, automatically set based on the label.
        kwargs: Additional parameters required for the metric, automatically set based on the label.
        direction: The direction for ranking of the metric,  "max" for maximization and "min" for minimization, automatically set based on the label.
        y_type: The type of predictions expected by the metric interface, automatically set based on the label.
    """

    label: MetricLabel
    config: GroupedMetricConfig | None = None
    custom_name: str | None = Field(None, exclude=True, alias="name")

    # Frozen metadata
    fn: Callable = Field(frozen=True, exclude=True)
    is_multitask: bool = Field(False, frozen=True, exclude=True)
    kwargs: dict = Field(default_factory=dict, frozen=True, exclude=True)
    direction: DirectionType = Field(frozen=True, exclude=True)
    y_type: PredictionKwargs = Field("y_pred", frozen=True, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def populate_metric_info(cls, data: Any) -> Any:
        if isinstance(data, dict):
            label = data.get("label")
            metric_info = DEFAULT_METRICS.get(label)
            if metric_info is None:
                raise ValueError(f"Unknown metric label: {label}")
            data.update(metric_info)
        return data

    @field_validator("is_multitask")
    @classmethod
    def disable_multitask_metrics(cls, value):
        if value:
            raise NotImplementedError(
                "Multitask metrics are not yet supported and will require non-obvious changes to the Metric interface."
            )

    @computed_field
    @property
    def kind(self) -> MetricKind:
        return "default" if self.config is None else self.config._kind

    @computed_field
    @property
    def name(self) -> str:
        if self.custom_name is not None:
            return self.custom_name

        name = self.label
        if self.kind != "default":
            name += "_" + self.kind
        return name

    def _default_score(
        self,
        y_true: GroundTruth,
        y_pred: BenchmarkPredictions | None = None,
        y_prob: BenchmarkPredictions | None = None,
    ) -> float:
        """Compute the metric.

        Args:
            y_true: The true target values.
            y_pred: The predicted target values, if any.
            y_prob: The predicted target probabilities, if any.
        """
        pred = prepare_predictions(y_pred, y_prob, self.y_type)
        kwargs = prepare_metric_kwargs(
            y_true.as_array("y"),
            pred.flatten(),
            self.y_type,
        )
        return self.fn(**kwargs, **self.kwargs)

    def score(
        self,
        y_true: GroundTruth,
        y_pred: BenchmarkPredictions | None = None,
        y_prob: BenchmarkPredictions | None = None,
    ) -> float:
        """Compute the metric.

        Args:
            y_true: The true target values.
            y_pred: The predicted target values, if any.
            y_prob: The predicted target probabilities, if any.
        """
        if self.kind == "default":
            return self._default_score(y_true, y_pred, y_prob)
        return self.config.score(y_true, y_pred, y_prob, DEFAULT_METRICS[self.label])

    def __call__(
        self,
        y_true: GroundTruth,
        y_pred: BenchmarkPredictions | None = None,
        y_prob: BenchmarkPredictions | None = None,
    ) -> float:
        """
        For convenience, calling a `Metric` is the same as calling its `score` method.

        ```python
        metric = Metric(label="mean_absolute_error")
        assert metric.score(y_true=first, y_pred=second) == metric(y_true=first, y_pred=second)
        ```
        """
        return self.score(y_true, y_pred, y_prob)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        match other:
            case Metric():
                return hash(other) == hash(self)
            case _:
                return False

    def __str__(self) -> str:
        return self.label + (
            json.dumps(self.config.model_dump(mode="json"), sort_keys=True) if self.config else ""
        )
