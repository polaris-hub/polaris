from typing import List
from typing import Optional

import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling

from .clusters import assign_using_clusters


class ClusterSampling(Sampling):
    """cluster sampling implementation for SIMPD"""

    def __init__(
        self,
        distance_threshold: float = 0.65,
        cluster_size_threshold: int = -1,
        clusters: Optional[List[List[int]]] = None,
    ):
        super().__init__()

        self._distance_threshold = distance_threshold
        self._cluster_size_threshold = cluster_size_threshold
        self._clusters = clusters

    def _do(self, problem, n_samples, **kwargs):
        if self._cluster_size_threshold > 0:
            cluster_size_threshold = self._cluster_size_threshold
        else:
            cluster_size_threshold = max(5, len(problem.distance_matrix) / 50)

        assignments = assign_using_clusters(
            problem.distance_matrix,
            self._distance_threshold,
            n_test=problem.n_max,
            cluster_size_threshold=cluster_size_threshold,
            n_samples=n_samples,
            clusters=self._clusters,
        )
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            assignments_indices = assignments[k]
            X[k, assignments_indices] = True

        return X


class SIMPDBinaryCrossover(Crossover):
    """crossover implementation for SIMPD

    This is adapted from the pymoo documentation
    """

    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            assignments_indices = np.where(np.logical_xor(p1, p2))[0]

            S = assignments_indices[np.random.permutation(len(assignments_indices))][:n_remaining]
            _X[0, k, S] = True

        return _X


class SIMPDMutation(Mutation):
    """mutation implementation for SIMPD

    This is adapted from the pymoo documentation
    """

    def __init__(self, swap_fraction: float = 0.1):
        super().__init__()
        self.swap_fraction = swap_fraction

    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false, size=int(self.swap_fraction * problem.n_max))] = True
            X[i, np.random.choice(is_true, size=int(self.swap_fraction * problem.n_max))] = False

        return X
