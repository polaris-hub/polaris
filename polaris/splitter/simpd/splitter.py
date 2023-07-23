from typing import Literal
from typing import Tuple
from typing import Optional

import numpy as np
import pandas as pd

from loguru import logger

from sklearn.model_selection import BaseShuffleSplit

from .simpd import run_SIMPD
from .simpd import SIMPDResult


class SIMPDSplitter(BaseShuffleSplit):
    """
    The SIMPD (SImulated Medicinal chemistry Project Data) is based on a multi-objective
    genetic algorithm (MOGA) to split a set of compounds with bioactivity data into one or more training and
    test sets that differ from each other in ways resembling the differences between the temporal training/test
    splits observed in medicinal chemistry projects.

    It's the implementation proposed in "SIMPD: an Algorithm for Generating Simulated Time Splits for Validating Machine Learning Approaches"
    available at <https://chemrxiv.org/engage/chemrxiv/article-details/6406049e6642bf8c8f10e189>.

    The source code has been largely inspired by the original authors implementation available at <https://github.com/rinikerlab/molecular_time_series/tree/55eb420ab0319fbb18cc00fe62a872ac568ad7f5>.
    """

    def __init__(
        self,
        n_splits: int = 1,
        # GA parameters
        pop_size: int = 500,
        ngens: int = 100,
        swap_fraction: float = 0.1,
        # SIMPD objectives and constraints
        simpd_descriptors: Optional[pd.DataFrame] = None,
        target_train_frac_active: float = -1,
        target_test_frac_active: float = -1,
        target_test_set_frac: float = 0.2,
        target_delta_test_frac_active: Optional[float] = None,
        target_GF_delta_window: Tuple[int, int] = (10, 30),
        target_G_val: int = 70,
        max_population_cluster_entropy: float = 0.9,
        pareto_weight_GF_delta: float = 10,
        pareto_weight_G: float = 5,
        # Misc
        num_threads: int = 1,
        random_seed: Optional[int] = 19,
        verbose: bool = True,
        verbose_pymoo: bool = True,
        progress: bool = True,
        progress_leave: bool = False,
    ):
        """
        Creates the splitter object.

        We invite the user to refer to the original paper for more details on the parameters.

        Args:
            n_splits: Number of splits to generate.
            pop_size: The population size for the GA.
            ngens: The number of generations for the GA.
            swap_fraction: The swap fraction for the GA. Swap N% of the bits in each mutation.
            simpd_descriptors: The descriptors to use for the GA. If None, the default descriptors from the paper will be used.
                Load them from `polaris.splitter.simpd.DEFAULT_SIMPD_DESCRIPTORS`.
            target_train_frac_active: The target fraction of active compounds in the training set. Set to -1 to disable.
            target_test_frac_active: The target fraction of active compounds in the test set. Set to -1 to disable.
            target_test_set_frac: The target fraction of the test set.
            target_delta_test_frac_active: The target delta of active between the test and training set.
            target_GF_delta_window: The target window for the GF delta.
            target_G_val: The target G value.
            max_population_cluster_entropy: The maximum cluster entropy.
            pareto_weight_GF_delta: The weight for the GF delta.
            pareto_weight_G: The weight for the G value.
            num_threads: The number of threads to use for the GA.
            random_seed: The random seed to use for the GA.
            verbose: Whether to print information about the splitter.
            verbose_pymoo: Whether to print information about the GA.
            progress: Whether to display a progress bar.
            progress_leave: Whether to leave the progress bar after completion.
        """
        super().__init__()

        self.n_splits = n_splits
        self.pop_size = pop_size
        self.ngens = ngens
        self.swap_fraction = swap_fraction
        self.simpd_descriptors = simpd_descriptors
        self.target_train_frac_active = target_train_frac_active
        self.target_test_frac_active = target_test_frac_active
        self.target_test_set_frac = target_test_set_frac
        self.target_delta_test_frac_active = target_delta_test_frac_active
        self.target_GF_delta_window = target_GF_delta_window
        self.target_G_val = target_G_val
        self.max_population_cluster_entropy = max_population_cluster_entropy
        self.pareto_weight_GF_delta = pareto_weight_GF_delta
        self.pareto_weight_G = pareto_weight_G
        self.num_threads = num_threads
        self.random_seed = random_seed
        self.verbose = verbose
        self.verbose_pymoo = verbose_pymoo
        self.progress = progress
        self.progress_leave = progress_leave

        # The attributes holding the computed splits + other informations the user might want to access
        self.simpd_splits: Optional[list] = None
        self.simpd_results: Optional[SIMPDResult] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ):
        """Fit the splitter against a given dataset.

        Args:
            X: An array of molecules.
            y: An array of activities (only 1D is supported).
            groups: An array of groups.

        """

        # Create a dataframe
        data = pd.DataFrame({"mol": X, "activivity": y})

        # Prepare the arguments
        args = {}

        args["data"] = data
        args["mol_column"] = "mol"
        args["activity_column"] = "activivity"

        # GA parameters
        args["pop_size"] = self.pop_size
        args["ngens"] = self.ngens
        args["swap_fraction"] = self.swap_fraction

        # SIMPD objectives and constraints
        args["simpd_descriptors"] = self.simpd_descriptors
        args["target_train_frac_active"] = self.target_train_frac_active
        args["target_test_frac_active"] = self.target_test_frac_active
        args["target_test_set_frac"] = self.target_test_set_frac
        args["target_delta_test_frac_active"] = self.target_delta_test_frac_active
        args["target_GF_delta_window"] = self.target_GF_delta_window
        args["target_G_val"] = self.target_G_val
        args["max_population_cluster_entropy"] = self.max_population_cluster_entropy
        args["pareto_weight_GF_delta"] = self.pareto_weight_GF_delta
        args["pareto_weight_G"] = self.pareto_weight_G

        # Misc
        args["num_threads"] = self.num_threads
        args["random_seed"] = self.random_seed
        args["verbose"] = self.verbose
        args["verbose_pymoo"] = self.verbose_pymoo
        args["progress"] = self.progress
        args["progress_leave"] = self.progress_leave

        self.simpd_results = run_SIMPD(**args)
        return self.simpd_results

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices"""
        if self.simpd_results is None:
            raise RuntimeError("The splitter has not be fitted yet")

        all_train_indices = self.simpd_results.train_indices[: self.n_splits]
        all_test_indices = self.simpd_results.test_indices[: self.n_splits]

        for train_indices, test_indices in zip(all_train_indices, all_test_indices):
            yield train_indices, test_indices
