import pytest
from polaris.splitter import ScaffoldSplit
from polaris.splitter._scaffold_split import get_scaffold


def test_get_scaffold(manual_smiles):
    scf = [get_scaffold(smi, make_generic=False) for smi in manual_smiles]
    assert scf == [""] + ["c1ccccc1"] * 5 + ["O=c1[nH]c(=O)c2[nH]cnc2[nH]1"] * 2


def test_get_scaffold_generic(manual_smiles):
    scf = [get_scaffold(smi, make_generic=True) for smi in manual_smiles]
    assert scf == [""] + ["*1:*:*:*:*:*:1"] * 5 + ["*=*1:*:*(=*):*2:*:*:*:*:2:*:1"] * 2


@pytest.mark.parametrize("make_generic", [True, False])
def test_splits_scaffold(dataset_smiles, make_generic):
    splitter = ScaffoldSplit(dataset_smiles, n_splits=2, make_generic=make_generic)
    for train_ind, test_ind in splitter.split(dataset_smiles):
        assert len(train_ind) + len(test_ind) == len(dataset_smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0

        train_scfs = set([get_scaffold(dataset_smiles[i], make_generic=make_generic) for i in train_ind])
        test_scfs = [get_scaffold(dataset_smiles[i], make_generic=make_generic) for i in test_ind]
        assert not any(test_scf in train_scfs for test_scf in test_scfs)
