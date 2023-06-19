import pytest
import datamol as dm
from polaris.splitter import MolecularWeightSplit


@pytest.mark.parametrize("generalize_to_larger", [True, False])
def test_splits_molecular_weight(dataset_smiles, generalize_to_larger):
    splitter = MolecularWeightSplit(generalize_to_larger=generalize_to_larger, n_splits=2)

    for train_ind, test_ind in splitter.split(dataset_smiles):
        assert len(train_ind) + len(test_ind) == len(dataset_smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > len(test_ind)
        assert len(train_ind) > 0 and len(test_ind) > 0

        train_mws = [dm.descriptors.mw(dm.to_mol(smi)) for smi in dataset_smiles[train_ind]]
        if generalize_to_larger:
            assert all(dm.descriptors.mw(dm.to_mol(smi)) > max(train_mws) for smi in dataset_smiles[test_ind])
        else:
            assert all(dm.descriptors.mw(dm.to_mol(smi)) < min(train_mws) for smi in dataset_smiles[test_ind])
