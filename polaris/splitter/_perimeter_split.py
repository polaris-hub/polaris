import numpy as np
from polaris.splitter._distance_split_base import KMeansReducedDistanceSplitBase


class PerimeterSplit(KMeansReducedDistanceSplitBase):
    """
    Places the pairs of data points with maximal pairwise distance in the test set.
    This was originally called the extrapolation-oriented split, introduced in  Szántai-Kis et. al., 2003
    """

    def get_split_from_distance_matrix(
        self, mat: np.ndarray, group_indices: np.ndarray, n_train: int, n_test: int
    ):
        """
        Iteratively places the pairs of data points with maximal pairwise distance in the test set.
        Anything that remains is added to the train set.

        Intuitively, this leads to a test set where all the datapoints are on the "perimeter"
        of the high-dimensional data cloud.
        """
        groups_set = np.unique(group_indices)

        # Sort the distance matrix to find the groups that are the furthest away from one another
        tril_indices = np.tril_indices_from(mat, k=-1)
        maximum_distance_indices = np.argsort(mat[tril_indices])[::-1]

        test_indices = []
        remaining = set(groups_set)

        for pos in maximum_distance_indices:
            if len(test_indices) >= n_test:
                break

            i, j = (
                tril_indices[0][pos],
                tril_indices[1][pos],
            )

            # If one of the molecules in this pair is already in the test set, skip to the next
            if not (i in remaining and j in remaining):
                continue

            remaining.remove(i)
            test_indices.extend(list(np.flatnonzero(group_indices == groups_set[i])))
            remaining.remove(j)
            test_indices.extend(list(np.flatnonzero(group_indices == groups_set[j])))

        train_indices = []
        for i in remaining:
            train_indices.extend(list(np.flatnonzero(group_indices == groups_set[i])))

        return np.array(train_indices), np.array(test_indices)
