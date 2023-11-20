import datamol as dm
import numpy as np
import pandas as pd
import pydantic
import pytest

from polaris.curation import run_chemistry_curation
from polaris.curation._chemistry_curator import _num_stereo_centers
from polaris.curation._data_curator import (
    _class_conversion,
    _identify_stereoisomers_with_activity_cliff,
    _merge_duplicates,
    check_outliers,
)
from polaris.curation.utils import Discretizer, LabelOrder, discretizer, outlier_detection


def test_discretizer():
    X = [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
    thresholds_binary = [0.5]
    thresholds_multiclass = [0, 1]

    values_binary = discretizer(X=X, thresholds=thresholds_binary)
    assert np.array_equal(values_binary, np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]]))

    values_binary_r = discretizer(X=X, thresholds=thresholds_binary, label_order=LabelOrder.desc.value)
    assert np.array_equal(values_binary_r, np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]]))

    values_multiclass = discretizer(X=X, thresholds=thresholds_multiclass)
    assert np.array_equal(values_multiclass, np.array([[2, 0, 2], [2, 1, 1], [1, 2, 0]]))

    values_multiclass_r = discretizer(
        X=X, thresholds=thresholds_multiclass, label_order=LabelOrder.desc.value
    )
    assert np.array_equal(values_multiclass_r, np.array([[0, 2, 0], [0, 1, 1], [1, 0, 2]]))

    with pytest.raises(pydantic.ValidationError):
        Discretizer(thresholds=thresholds_multiclass, label_order="WrongType")

    with pytest.raises(ValueError):
        discretizer(X=X, thresholds=thresholds_multiclass, label_order="WrongType")


def test_class_conversion():
    colnames = ["data_col_1", "data_col_2", "data_col_3"]
    data = pd.DataFrame.from_dict(
        {colnames[0]: [1.0, -1.0, 2.0], colnames[1]: [2.0, 0.0, 0.0], colnames[2]: [0.0, 1.0, -1.0]}
    )
    converted = _class_conversion(
        data,
        data_cols=colnames,
        conversion_params={
            "data_col_1": {"thresholds": [0, 1], "label_order": "ascending"},
            "data_col_2": {"thresholds": [0.5], "label_order": LabelOrder.acs.value},
            "data_col_3": {"thresholds": [0, 1], "label_order": LabelOrder.desc.value},
        },
    )
    assert converted["CLASS_data_col_1"].tolist() == [2, 0, 2]
    assert converted["CLASS_data_col_2"].tolist() == [1, 0, 0]
    assert converted["CLASS_data_col_3"].tolist() == [1, 0, 2]


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


def test_check_outlier_zscore():
    data = pd.DataFrame(np.random.normal(0, 0.1, size=(100, 1)), columns=["data_col"])
    # create 5 data points which are out of distribution
    num_outlier = 5
    data.loc[: num_outlier - 1, "data_col"] = 10
    outilers = outlier_detection(X=data[["data_col"]].values, method="zscore")
    assert len(outilers) == num_outlier

    data_outlier = check_outliers(data, ["data_col"])
    assert data_outlier["OUTLIER_data_col"].sum() == num_outlier


def test_identify_stereoisomers_with_activity_cliff():
    vals = list(np.random.randint(0, 10, 50)) * 2 + np.random.normal(0, 0.01, 100)
    groups = list(range(50)) * 2
    num_cliff = 10
    index_cliff = np.random.randint(0, 50, num_cliff)
    vals[index_cliff] = 1000

    data = pd.DataFrame({"data_col": vals, "groupby_col": groups})
    ids = _identify_stereoisomers_with_activity_cliff(
        data=data, data_col="data_col", groupby_col="groupby_col", threshold=1
    )
    # check if identifed ids are correct
    assert set(ids) == set(index_cliff)


def test_merge_duplicates():
    num_sample = 100
    ids = np.array(range(num_sample))
    max_ind = max(ids)
    num_dup = 5
    for index in range(num_dup):
        ids[max_ind - index] = index
    data_col_1 = np.random.normal(1, 0.01, 100)
    data_col_2 = np.random.normal(2, 0.01, 100)
    data_col_3 = np.random.normal(3, 0.01, 100)
    data = pd.DataFrame(
        {"data_col_1": data_col_1, "data_col_2": data_col_2, "data_col_3": data_col_3, "ids": ids}
    )

    merged = _merge_duplicates(
        data=data, data_cols=["data_col_1", "data_col_2", "data_col_3"], merge_on=["ids"]
    )

    # check the data points been merged
    assert data.shape[0] == merged.shape[0] + num_dup
    # check the merged values are correct
    for index in range(num_dup):
        assert data.loc[[index, max_ind - index], "data_col_1"].median() == merged.loc[index, "data_col_1"]
        assert data.loc[[index, max_ind - index], "data_col_2"].median() == merged.loc[index, "data_col_2"]
        assert data.loc[[index, max_ind - index], "data_col_3"].median() == merged.loc[index, "data_col_3"]


def test_num_undefined_stereo_centers():
    # mol with no stereo centers
    mol = dm.to_mol("CCCC")
    num_all, num_defined, num_undefined = _num_stereo_centers(mol)
    assert num_all == 0
    assert num_defined == 0
    assert num_undefined == 0

    # mol with all defined centers
    mol = dm.to_mol("C1C[C@H](C)[C@H](C)[C@H](C)C1")
    num_all, num_defined, num_undefined = _num_stereo_centers(mol)
    assert num_all == 3
    assert num_defined == 3
    assert num_undefined == 0

    # mol with partial defined centers
    mol = dm.to_mol("C[C@H](F)C(F)(Cl)Br")
    num_all, num_defined, num_undefined = _num_stereo_centers(mol)
    assert num_all == 2
    assert num_defined == 1
    assert num_undefined == 1

    # mol with no defined centers
    mol = dm.to_mol("CC(F)C(F)(Cl)Br")
    num_all, num_defined, num_undefined = _num_stereo_centers(mol)
    assert num_all == 2
    assert num_defined == 0
    assert num_undefined == 2
