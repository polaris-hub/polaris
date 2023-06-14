import datamol as dm
import numpy as np
from sklearn.model_selection import ShuffleSplit
from polaris.splitter import (
    MOODSplitter,
    ScaffoldSplit,
    PerimeterSplit,
    MaxDissimilaritySplit,
)


def test_splits_integration():
    dataset_mols = dm.data.freesolv(as_df=False)
    dataset_smiles = [dm.to_smiles(mol) for mol in dataset_mols if mol is not None]

    deployment_mols = dm.data.solubility(as_df=False)
    deployment_smiles = [dm.to_smiles(mol) for mol in deployment_mols if mol is not None]

    feats_dataset = np.array([dm.to_fp(mol) for mol in dataset_smiles])
    feats_deployment = np.array([dm.to_fp(mol) for mol in deployment_smiles])

    splitters = {
        "random": ShuffleSplit(),
        "scaffold": ScaffoldSplit(dataset_smiles),
        "perimeter": PerimeterSplit(metric="jaccard", n_clusters=5),
        "max_dissimilarity": MaxDissimilaritySplit(metric="jaccard", n_clusters=5),
    }

    splitter = MOODSplitter(splitters, metric="jaccard")
    splitter.fit(X=feats_dataset, X_deployment=feats_deployment)

    for train_ind, test_ind in splitter.split(feats_dataset):
        train = feats_dataset[train_ind]
        test = feats_dataset[test_ind]

        assert len(train) > 0 and len(test) > 0
        assert len(train) + len(test) == len(feats_dataset)
