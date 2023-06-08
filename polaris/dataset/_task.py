from typing import Union, List

from polaris.dataset import Dataset, Modality
from polaris.evaluate import Metric
from polaris.utils.exceptions import InvalidTaskError


class Task:
    """A task wraps a dataset with some additional information to form a machine learning task."""

    def __init__(
        self,
        dataset: Dataset,
        target_cols: Union[List[str], str],
        input_cols: Union[List[str], str],
        split: List[List[int]],
        metrics: List[Union[Metric, str]],
    ):
        self.dataset = dataset
        self.target_cols = self._verify_cols(target_cols, Modality.TARGET, inclusive_filter=True)
        self.input_cols = self._verify_cols(input_cols, Modality.TARGET, inclusive_filter=False)
        self._metrics = self._verify_metrics(metrics)
        self._split = self._verify_split(split)

    def _verify_cols(
        self,
        cols: List[str],
        modality_filter: Union[Modality, List[Modality]],
        inclusive_filter: bool = True,
    ):
        if not isinstance(cols, List):
            cols = [cols]
        if not isinstance(modality_filter, List):
            modality_filter = [modality_filter]
        if not all(c in self.dataset.table.columns for c in cols):
            raise InvalidTaskError(f"Not all specified target columns were found in the dataset.")
        if not inclusive_filter:
            modality_filter = list(set(Modality) - set(modality_filter))
        if not all(self.dataset.info.modalities[c] in modality_filter for c in cols):
            raise InvalidTaskError(f"Not all input specified columns have the correct modality.")
        return cols

    def _verify_metrics(self, metrics: List[Union[Metric, str]]):  # noqa
        if not isinstance(metrics, List):
            metrics = [metrics]
        metrics = [m if isinstance(m, Metric) else Metric.get_by_name(m) for m in metrics]
        return metrics

    def _verify_split(self, split: List[List[int]]):  # noqa
        """
        TODO:
         - What about multi-task splits? E.g. one mol can be in both the train and test set, with different readouts.
        """

        return split

    def prepare(self):
        """
        TODO:
          - Should we filter out NaN values? E.g. we could have a sparse multi-task dataset.
          - On the above; How to specify / verify indices in that case?
        """
        pass

    def split(self):
        pass

    def evaluate(self):
        pass
