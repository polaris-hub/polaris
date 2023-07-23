from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from pymoo.core.problem import ElementwiseProblem


from .distance import modified_spatial_stats_dmat
from .clusters import population_cluster_entropy


class SplitProblem_DescriptorAndFGDeltas(ElementwiseProblem):
    """This is the pymoo Problem form used for SIMPD

    From https://github.com/rinikerlab/molecular_time_series/blob/55eb420ab0319fbb18cc00fe62a872ac568ad7f5/ga_lib_3.py#L966
    """

    def __init__(
        self,
        binned_activities: np.ndarray,
        fps: np.ndarray,
        distance_matrix: np.ndarray,
        descriptor_values: np.ndarray,
        descriptor_delta_targets: np.ndarray,
        n_max: int,
        clusters: Optional[List[List[int]]] = None,
        target_train_frac_active: float = -1,
        target_test_frac_active: float = -1,
        target_delta_test_frac_active: Optional[float] = None,
        target_GF_delta_window: Tuple[int, int] = (10, 30),
        target_G_val: int = 70,
        max_population_cluster_entropy: float = 0.9,
        random_seed: Optional[int] = 19,
        **kwargs,
    ):
        assert len(binned_activities) == len(fps)
        assert len(fps) == len(descriptor_values)

        self.activities = np.array(binned_activities)
        self.descriptor_values = np.array(descriptor_values)
        self.descriptor_delta_targets = np.array(descriptor_delta_targets)

        assert self.descriptor_values.shape[1] == self.descriptor_delta_targets.shape[0]

        self.fps = fps
        self.distance_matrix = distance_matrix
        self.max_population_cluster_entropy = max_population_cluster_entropy
        self.clusters = clusters

        self._nObjs = len(descriptor_values[0])

        if target_test_frac_active > 0 or target_train_frac_active > 0:
            self.tgtTestFrac = target_test_frac_active
            self.tgtTrainFrac = target_train_frac_active
            self._nObjs += int(target_test_frac_active > 0) + int(target_train_frac_active > 0)
            self.tgtFrac = None
            self.delta_test_frac_active = None

        elif target_delta_test_frac_active is not None:
            self.delta_test_frac_active = target_delta_test_frac_active
            self.tgtFrac = None
            self._nObjs += 1

        else:
            self._nObjs += 1
            self.tgtFrac = np.sum(self.activities == 1) / len(self.activities)
            self.delta_test_frac_active = None

        self.target_GF_delta_window = target_GF_delta_window
        self._nObjs += 1
        self.target_G_val = target_G_val
        self._nObjs += 1

        n_ieq_constr = 1
        if self.clusters is not None:
            n_ieq_constr = 2

        super().__init__(
            n_var=len(descriptor_values),
            n_obj=self._nObjs,
            n_ieq_constr=n_ieq_constr,
            **kwargs,
        )
        self.n_max = n_max

        self.random_seed = random_seed

    def _evaluate(self, x, out, *args, **kwargs):
        train = np.median(self.descriptor_values[~x], axis=0)
        test = np.median(self.descriptor_values[x], axis=0)
        descr_deltas = test - train

        descr_objects = abs(descr_deltas - self.descriptor_delta_targets)
        objectives = list(descr_objects)
        # objectives = []

        train_acts = self.activities[~x]
        train_frac = np.sum(train_acts, axis=0) / len(train_acts)
        test_acts = self.activities[x]
        test_frac = np.sum(test_acts, axis=0) / len(test_acts)

        if self.tgtFrac is not None:
            objectives.append(abs(test_frac - self.tgtFrac))

        elif self.delta_test_frac_active is not None:
            dTestFracActive = test_frac - np.sum(self.activities, axis=0) / len(self.activities)
            objectives.append(abs(self.delta_test_frac_active - dTestFracActive))

        else:
            if self.tgtTrainFrac > 0:
                objectives.append(abs(train_frac - self.tgtTrainFrac))
            if self.tgtTestFrac > 0:
                objectives.append(abs(test_frac - self.tgtTestFrac))

        allIdx = np.arange(0, len(x), dtype=int)
        testIdx = allIdx[x]
        trainIdx = allIdx[~x]
        vals, g_vals, f_vals, s_vals = modified_spatial_stats_dmat(
            self.distance_matrix,
            testIdx,
            trainIdx,
            include_test_in_bootstrap=False,
            random_seed=self.random_seed,
        )
        sum_F = np.sum(f_vals)
        sum_G = np.sum(g_vals)
        delt = sum_G - sum_F

        if delt > self.target_GF_delta_window[1]:
            objectives.append(delt - self.target_GF_delta_window[1])
        elif delt < self.target_GF_delta_window[0]:
            objectives.append(self.target_GF_delta_window[0] - delt)
        else:
            objectives.append(0)

        if sum_G < self.target_G_val:
            objectives.append(self.target_G_val - sum_G)
        else:
            objectives.append(0)

        # objectives
        out["F"] = objectives

        # constraints
        out["G"] = [(self.n_max - np.sum(x)) ** 2]

        if self.clusters is not None:
            # keep the entropy below 0.9
            out["G"].append(
                population_cluster_entropy(x, self.clusters) - self.max_population_cluster_entropy
            )
