from loguru import logger
from sklearn.model_selection import GroupShuffleSplit


class PredefinedGroupShuffleSplit(GroupShuffleSplit):
    """
    Simple class to overcome the limitation of the MOODSplitter
    that all splitters need to use the same grouping.
    """

    def __init__(self, groups, n_splits=10, *, test_size=None, train_size=None, random_state=None):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._groups = groups

    def _iter_indices(self, X=None, y=None, groups=None):
        """Generate (train, test) indices"""
        if groups is not None:
            logger.warning("Ignoring the groups parameter in favor of the predefined groups")
        yield from super()._iter_indices(X, y, self._groups)
