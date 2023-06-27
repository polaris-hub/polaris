import numpy as np

from polaris.splitter import KMeansSplit


def test_splits_kmeans_default_feats(dataset_smiles):
    splitter = KMeansSplit(n_splits=2)

    for train_ind, test_ind in splitter.split(dataset_smiles):
        assert len(train_ind) + len(test_ind) == len(dataset_smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0
        assert splitter._cluster_metric == "jaccard"


def test_splits_kmeans():
    X = np.random.random((100, 100))
    splitter = KMeansSplit(n_splits=2, metric="euclidean")

    for train_ind, test_ind in splitter.split(X):
        assert len(train_ind) + len(test_ind) == len(X)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0
        assert splitter._cluster_metric == "euclidean"
