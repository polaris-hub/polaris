from typing import Optional

import random

import numpy as np


def modified_spatial_stats_dmat(
    dmat: np.ndarray,
    test_indices: np.ndarray,
    train_indices: np.ndarray,
    vals: np.ndarray = np.arange(0, 1.01, 0.01),
    include_test_in_bootstrap: bool = True,
    random_seed: Optional[int] = 19,
):
    """Calculates F using closest member of train to test instead of vice-versa."""

    rng = random.Random(random_seed)

    g_vals = get_distance_cdf_dmat(
        dmat,
        test_indices,
        test_indices,
        remove_self=True,
        vals=vals,
    )
    tidx = list(train_indices)
    if include_test_in_bootstrap:
        tidx += list(test_indices)

    bootstrap = [tidx[x] for x in [rng.randint(0, len(tidx) - 1) for x in range(len(tidx))]]
    bootstrap = np.array(bootstrap)
    f_vals = get_distance_cdf_dmat(
        dmat,
        test_indices,
        bootstrap,
        remove_self=False,
        vals=vals,
    )
    s_vals = [f - g for f, g in zip(f_vals, g_vals)]
    return vals, g_vals, f_vals, s_vals


def get_distance_cdf_dmat(
    dmat: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    remove_self: bool = False,
    vals=np.arange(0, 1.01, 0.01),
):
    """CDF of number of points in fps2 which are less than distThresh
    from each point in fps1.
    """
    dmat = dmat[idx1][:, idx2]
    n_points = len(idx1)
    if remove_self:
        np.fill_diagonal(dmat, np.nan)
    dmat = np.nanmin(dmat, axis=1)

    res = []
    for v in vals:
        res.append(np.sum(dmat <= v) / n_points)
    return res
