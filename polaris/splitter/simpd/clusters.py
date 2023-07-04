from typing import Literal
from typing import List
from typing import Optional

import random

import numpy as np

from rdkit.ML.InfoTheory import rdInfoTheory  # type: ignore
from rdkit.ML.Cluster import Butina


def cluster_data(
    dmat: np.ndarray,
    distance_threshold: float,
    cluster_size_threshold: float,
    combine_random: bool = False,
    random_seed: Optional[int] = 19,
):
    """From https://github.com/rinikerlab/molecular_time_series/blob/55eb420ab0319fbb18cc00fe62a872ac568ad7f5/ga_lib_3.py#L1152"""

    rng = random.Random(random_seed)

    nfps = len(dmat)
    symmDmat = []
    for i in range(1, nfps):
        symmDmat.extend(dmat[i, :i])

    cs = Butina.ClusterData(
        symmDmat,
        nfps,
        distance_threshold,
        isDistData=True,
        reordering=True,
    )
    cs = sorted(cs, key=lambda x: len(x), reverse=True)

    # start with the large clusters
    largeClusters = [list(c) for c in cs if len(c) >= cluster_size_threshold]
    if not largeClusters:
        raise ValueError("no clusters found")

    # now combine the small clusters to make larger ones
    if combine_random:
        tmpCluster = []
        for c in cs:
            if len(c) >= cluster_size_threshold:
                continue
            tmpCluster.extend(c)
            if len(tmpCluster) >= cluster_size_threshold:
                rng.shuffle(tmpCluster)
                largeClusters.append(tmpCluster)
                tmpCluster = []
        if tmpCluster:
            largeClusters.append(tmpCluster)
    else:
        # add points from small clusters to the nearest larger cluster
        # nearest is defined by the nearest neighbor in that cluster
        for c in cs:
            if len(c) >= cluster_size_threshold:
                continue
            for idx in c:
                closest = -1
                minD = 1e5
                for cidx, clust in enumerate(largeClusters):
                    for elem in clust:
                        d = dmat[idx, elem]
                        if d < minD:
                            closest = cidx
                            minD = d
                assert closest > -1
                largeClusters[closest].append(idx)

    return largeClusters


def population_cluster_entropy(X, clusters: List[List[int]]):
    if len(clusters) <= 1:
        return 0

    ccounts = np.zeros(len(clusters), int)
    for i, clust in enumerate(clusters):
        for entry in clust:
            if X[entry]:
                ccounts[i] += 1

    term1 = rdInfoTheory.InfoEntropy(ccounts)
    term2 = rdInfoTheory.InfoEntropy(np.ones(len(clusters), int))
    return term1 / term2


def assign_using_clusters(
    dmat: np.ndarray,
    distance_threshold: float,
    n_test: int,
    cluster_size_threshold: float = 5,
    n_samples: int = 1,
    selection_strategy: Literal["DIVERSE", "CLUSTERS_SPLIT"] = "DIVERSE",
    combine_random: bool = False,
    clusters: Optional[List[List[int]]] = None,
    random_seed: Optional[int] = 19,
):
    if selection_strategy not in ["DIVERSE", "CLUSTERS_SPLIT"]:
        raise ValueError(
            f"Invalid selection strategy: {selection_strategy}. Valid values are 'DIVERSE' or 'CLUSTERS_SPLIT'."
        )

    if clusters is not None:
        large_clusters = clusters
    else:
        large_clusters = cluster_data(
            dmat=dmat,
            distance_threshold=distance_threshold,
            cluster_size_threshold=cluster_size_threshold,
            combine_random=combine_random,
            random_seed=random_seed,
        )

    rng = random.Random(random_seed)

    res = []
    for i in range(n_samples):
        if selection_strategy == "DIVERSE":
            ordered = []
            for c in large_clusters:
                # randomize the points in the cluster
                rng.shuffle(c)
                ordered.extend((i / len(c), x) for i, x in enumerate(c))
            ordered = [y for x, y in sorted(ordered)]
            test = ordered[:n_test]

        elif selection_strategy == "CLUSTERS_SPLIT":
            rng.shuffle(large_clusters)
            test = []
            for clus in large_clusters:
                nRequired = n_test - len(test)
                test.extend(clus[:nRequired])
                if len(test) >= n_test:
                    break
        else:
            raise ValueError(f"Invalid selection strategy: {selection_strategy}.")

        res.append(test)
    return res
