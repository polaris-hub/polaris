import pandas as pd
import pytest
import numpy as np

import datamol as dm
from polaris.curation.utils import discretizer
from polaris.curation import run_chemistry_curation
from polaris.curation.utils import outlier_detection, OUTLIER_METHOD


def test_discretizer():
    X = [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
    thresholds_binary = [0.5]
    thresholds_multiclass = [0, 1]

    values_binary = discretizer(X=X, thresholds=thresholds_binary)
    assert np.array_equal(values_binary, np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]]))

    values_binary_r = discretizer(X=X, thresholds=thresholds_binary, label_order="descending")
    assert np.array_equal(values_binary_r, np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]]))

    values_multiclass = discretizer(X=X, thresholds=thresholds_multiclass)
    assert np.array_equal(values_multiclass, np.array([[2, 0, 2], [2, 1, 1], [1, 2, 0]]))

    values_multiclass_r = discretizer(X=X, thresholds=thresholds_multiclass, label_order="descending")
    assert np.array_equal(values_multiclass_r, np.array([[0, 2, 0], [0, 1, 1], [1, 0, 2]]))


def test_run_chemistry_curation():
    mols = [
        "COc1ccc2ncc(=O)n(CCN3CC[C@H](NCc4ccc5c(n4)NC(=O)CO5)[C@H](O)C3)c2c1",
        "COc1ccc2ncc(=O)n(CCN3CC[C@@H](NCc4ccc5c(n4)NC(=O)CO5)[C@@H](O)C3)c2c1",
        "C[C@H]1CN(Cc2cc(Cl)ccc2OCC(=O)O)CCN1S(=O)(=O)c1ccccc1",
        "C[C@@H]1CN(Cc2cc(Cl)ccc2OCC(=O)O)CCN1S(=O)(=O)c1ccccc1",
        "CC[C@@H](c1ccc(C(=O)O)c(Oc2cccc(Cl)c2)c1)N1CCC[C@H](n2cc(C)c(=O)[nH]c2=O)C1",
        "CC[C@H](c1ccc(C(=O)O)c(Oc2cccc(Cl)c2)c1)N1CCC[C@H](n2cc(C)c(=O)[nH]c2=O)C1",
        "CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12",
        "CCOc1cc2nn(CCC(C)(C)O)cc2cc1NC(=O)c1cccc(C(F)F)n1",
        "CN(c1ncc(F)cn1)C1CCCNC1",
        "CC(C)(Oc1ccc(-c2cnc(N)c(-c3ccc(Cl)cc3)c2)cc1)C(=O)O",
        "CC(C)(O)CCn1cc2cc(NC(=O)c3cccc(C(F)(F)F)n3)c(C(C)(C)O)cc2n1",
        "CC#CC(=O)N[C@H]1CCCN(c2c(F)cc(C(N)=O)c3[nH]c(C)c(C)c23)C1",
        "COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",
        "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
        "CC(C)NC(=O)COc1cccc(-c2nc(Nc3ccc4[nH]ncc4c3)c3ccccc3n2)c1.[Na]",
        "CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1.Cl",
    ]
    # check stereoisomers are included.
    df = run_chemistry_curation(mols=mols)
    assert len(df.smiles.unique()) == len(mols)

    # check if stereoisomers are ignored
    df = run_chemistry_curation(mols=mols, ignore_stereo=True)
    assert len(df.smiles.unique()) == len(mols) - 3

    # check whether salts/solvents were removed.
    for smiles in df.smiles:
        mol = dm.to_mol(smiles)
        assert dm.same_mol(dm.remove_salts_solvents(mol), mol)


def test_outlier_detection_zscore():
    data = pd.DataFrame(np.random.normal(0, 0.1, size=(100, 1)), columns=["data_col"])
    # create 5 data points which are out of distribution
    num_outlier = 5
    data.loc[:num_outlier-1, "data_col"] = 100
    outilers = outlier_detection(X=data[["data_col"]].values, method="zscore")
    assert len(outilers) == num_outlier
