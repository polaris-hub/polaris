from typing import Tuple
from typing import Optional
from typing import cast
from typing import List

import dataclasses
import multiprocessing.pool

from loguru import logger

import pandas as pd
import numpy as np

from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.result import Result

from .clusters import cluster_data
from .preprocess import preprocess_SIMPD_mols
from .problem import SplitProblem_DescriptorAndFGDeltas
from .ga import ClusterSampling
from .ga import SIMPDBinaryCrossover
from .ga import SIMPDMutation
from .callbacks import ProgressCallback


@dataclasses.dataclass
class SIMPDResult:
    train_indices: List[np.ndarray]
    test_indices: List[np.ndarray]
    pareto_scores: np.ndarray
    pareto_sorted_indices: np.ndarray
    Xs: np.ndarray
    Fs: np.ndarray
    pymoo_results: Result


def run_SIMPD(
    data: pd.DataFrame,
    mol_column: str = "mol",
    activity_column: str = "active",
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
    # Preprocess the molecules
    descriptor_values, fps, descriptor_delta_targets, distance_matrix = preprocess_SIMPD_mols(
        data,
        mol_column=mol_column,
        activity_column=activity_column,
        simpd_descriptors=simpd_descriptors,
        verbose=verbose,
        progress=progress,
        progress_leave=progress_leave,
    )

    # Setup a runner for pymoo
    # NOTE(hadim): it's possible to use `multiprocessing.Pool` as well.
    if num_threads > 1:
        pool = multiprocessing.pool.ThreadPool(num_threads)
        runner = StarmapParallelization(pool.starmap)
    else:
        runner = None

    max_points = len(data)
    keep = max_points  # NOTE(hadim): I think this is just for dev purposes

    assert target_test_set_frac > 0 and target_test_set_frac < 1
    n_max = int(target_test_set_frac * keep)

    if verbose:
        logger.info(f"Working with {keep} points and picking {n_max}")

    # Clutering for the starting points
    distance_threshold = 0.65
    cluster_size_threshold = max(5, len(data) / 50)

    clusters = cluster_data(
        distance_matrix,
        distance_threshold,
        cluster_size_threshold=cluster_size_threshold,
        random_seed=random_seed,
    )

    cluster_sizes = [len(c) for c in clusters]

    if verbose:
        logger.info(
            f"Clustering the starting points with a distance threshold of {distance_threshold} and a cluster size threshold of {cluster_size_threshold}."
        )
        logger.info(f"{len(clusters)} clusters have been created of size: {cluster_sizes}")

    # Init the pymoo problem
    problem = SplitProblem_DescriptorAndFGDeltas(
        binned_activities=data[activity_column].to_numpy(),
        fps=fps,
        distance_matrix=distance_matrix,
        descriptor_values=descriptor_values,
        descriptor_delta_targets=descriptor_delta_targets,
        n_max=n_max,
        runner=runner,
        clusters=clusters,
        target_train_frac_active=target_train_frac_active,
        target_test_frac_active=target_test_frac_active,
        target_delta_test_frac_active=target_delta_test_frac_active,
        target_GF_delta_window=target_GF_delta_window,
        target_G_val=target_G_val,
        max_population_cluster_entropy=max_population_cluster_entropy,
        random_seed=random_seed,
    )

    # Init the sampling
    sampling = ClusterSampling(clusters=clusters)

    # Init the GA algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,  # type: ignore
        crossover=SIMPDBinaryCrossover(),  # type: ignore
        mutation=SIMPDMutation(swap_fraction=swap_fraction),  # type: ignore
        eliminate_duplicates=True,
    )

    if verbose:
        logger.info("Start the optimization.")

    # Setup a progress bar
    pbar_callback = ProgressCallback(
        n_gen=ngens,
        progress=progress,
        leave=progress_leave,
        auto_tqdm=True,
        description="Optimization",
    )

    # Run the minimzation procedure
    pymoo_results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=("n_gen", ngens),
        seed=random_seed,
        verbose=verbose_pymoo,
        callback=pbar_callback,
    )

    Xs = cast(np.ndarray, pymoo_results.X)  # type: ignore
    Fs = cast(np.ndarray, pymoo_results.F)  # type: ignore

    n_objectives = Fs.shape[1]
    n_solutions = len(Fs)

    if verbose:
        logger.info(f"Found {n_solutions} solutions")

    if verbose:
        logger.info("Scoring the solutions.")

    # Initialize equal weights for all the objectives
    weights = np.ones((n_objectives,), dtype=int)

    # We favour F and G (the spatial metrics)
    weights[-2] = pareto_weight_GF_delta
    weights[-1] = pareto_weight_G

    # Compute the scores (the higher the better)
    pareto_scores = score_pareto_solutions(Fs=Fs, weights=weights)
    pareto_sorted_indices = np.argsort(pareto_scores)[::-1]

    # Sort Xs and Fs by pareto scores
    Xs = Xs[pareto_sorted_indices]
    Fs = Fs[pareto_sorted_indices]
    pareto_scores = pareto_scores[pareto_sorted_indices]

    if verbose:
        logger.info(f"Objective values for the best solution: {Fs[0]}")

    # Extract the train/test indices from every solutions
    test_indices = []
    train_indices = []

    for i in range(n_solutions):
        solution = Xs[i]
        test_indices.append(np.where(solution)[0])
        train_indices.append(np.where(~solution)[0])

    results = SIMPDResult(
        train_indices=train_indices,
        test_indices=test_indices,
        pareto_scores=pareto_scores,
        Xs=Xs,
        Fs=Fs,
        pymoo_results=pymoo_results,
        pareto_sorted_indices=pareto_sorted_indices,
    )

    return results


def score_pareto_solutions(Fs: np.ndarray, weights: np.ndarray):
    Fs = np.copy(Fs)
    qs = np.quantile(Fs, 0.9, axis=0)
    maxv = np.max(np.abs(Fs), axis=0)
    for i, q in enumerate(qs):
        if q == 0:
            qs[i] = maxv[i]
            if qs[i] == 0:
                qs[i] = 1
    Fs /= qs
    Fs = np.exp(Fs * -1)
    weights = np.array(weights, float)
    # normalize
    weights /= np.sum(weights)

    Fs *= weights
    return np.sum(Fs, axis=1)
