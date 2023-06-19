import numpy as np
from polaris.splitter._base import KMeansReducedDistanceSplitBase


class MaxDissimilaritySplit(KMeansReducedDistanceSplitBase):
    """Splits the data such that the train and test set are maximally dissimilar."""

    def get_split_from_distance_matrix(
        self, mat: np.ndarray, group_indices: np.ndarray, n_train: int, n_test: int
    ):
        """
        The Maximum Dissimilarity Split splits the data by trying to maximize the distance between train and test.

        This is done as follows:
            (1) As initial test sample, take the data point that on average is furthest from all other samples.
            (2) As initial train sample, take the data point that is furthest from the initial test sample.
            (3) Iteratively add the train sample that is closest to the initial train sample.
        """

        n_samples = n_train + n_test
        groups_set = np.unique(group_indices)

        # The initial test cluster is the one with the
        # highest mean distance to all other clusters
        test_idx = np.argmax(mat.mean(axis=0))

        # The initial train cluster is the one furthest from
        # the initial test cluster
        train_idx = np.argmax(mat[test_idx])

        train_indices = np.flatnonzero(group_indices == groups_set[train_idx])
        test_indices = np.flatnonzero(group_indices == groups_set[test_idx])

        # Iteratively add the train cluster that is closest
        # to the _initial_ train cluster.
        sorted_groups = np.argsort(mat[train_idx])
        for group_idx in sorted_groups:
            if len(train_indices) >= n_train:
                break

            if group_idx == train_idx or group_idx == test_idx:
                continue

            indices_to_add = np.flatnonzero(group_indices == groups_set[group_idx])
            train_indices = np.concatenate([train_indices, indices_to_add])

        # Construct test set
        remaining_groups = list(set(range(n_samples)) - set(train_indices) - set(test_indices))
        test_indices = np.concatenate([test_indices, remaining_groups]).astype(int)

        return train_indices, test_indices
