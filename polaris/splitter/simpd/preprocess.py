from typing import Optional
import importlib

from loguru import logger

import pandas as pd
import datamol as dm
import numpy as np

from scipy.spatial import distance
from sklearn.utils.multiclass import type_of_target

from .descriptors import DEFAULT_SIMPD_DESCRIPTORS


def preprocess_SIMPD_mols(
    data: pd.DataFrame,
    mol_column: str = "mol",
    activity_column: str = "active",
    simpd_descriptors: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    progress: bool = True,
    progress_leave: bool = False,
):
    if simpd_descriptors is None:
        simpd_descriptors = DEFAULT_SIMPD_DESCRIPTORS
    else:
        if "function" not in simpd_descriptors.columns:
            raise ValueError("The simpd_descriptors must contains a 'function' column.")
        if "target_delta_value" not in simpd_descriptors.columns:
            raise ValueError("The simpd_descriptors must contains a 'target_delta_value' column.")

    # Convert descriptor functions to callable
    descriptor_functions = []
    for fn in simpd_descriptors["function"]:
        if isinstance(fn, str):
            module_name = ".".join(fn.split(".")[:-1])
            fn_name = fn.split(".")[-1]
            fn = getattr(importlib.import_module(module_name), fn_name)
        descriptor_functions.append(fn)

    # Preprocess the molecules
    def _preprocess_mol(mol):
        datum = {}

        # Compute descriptors
        datum["descriptor_values"] = [fn(mol) for fn in descriptor_functions]

        # Compute fingerprints
        datum["fp"] = dm.to_fp(mol, radius=3)

        return datum

    if verbose:
        logger.info("Compute descriptors and fingerprint values for the molecules.")

    mol_data = dm.parallelized(
        _preprocess_mol,
        data[mol_column],
        progress=progress,
        n_jobs=1,
        tqdm_kwargs=dict(leave=progress_leave, desc="Preprocess molecules"),
    )
    mol_data = pd.DataFrame(mol_data)

    # Retrieve descriptors and fp arrays
    descriptor_values = np.vstack(mol_data["descriptor_values"].values)  # type: ignore
    fps = np.vstack(mol_data["fp"].values)  # type: ignore

    # Retrieve the descriptor's targets
    descriptor_delta_targets = simpd_descriptors["target_delta_value"].to_numpy()

    # Compute the distance matrix
    # NOTE(hadim): possible to parallelize for very large dataset with
    # `sklearn.metrics.pairwise_distances_chunked` if needed.

    if verbose:
        logger.info("Compute the distance matrix for the molecules.")

    distance_matrix = distance.pdist(fps, metric="jaccard")
    distance_matrix = distance.squareform(distance_matrix, force="tomatrix")

    # Check the activity column only contains 0 or 1 values
    target_type = type_of_target(data[activity_column])
    if target_type != "binary":
        raise ValueError(
            f"The activity column '{activity_column}' must be of type 'binary', found '{target_type}'. Contains: {data[activity_column].values}."
        )

    return descriptor_values, fps, descriptor_delta_targets, distance_matrix
