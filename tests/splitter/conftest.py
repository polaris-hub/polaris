import pytest


@pytest.fixture(scope="module")
def manual_smiles():
    return [
        "CCCCC",
        "C1=CC=CC=C1",
        "CCCCOC(C1=CC=CC=C1)OCCCC",
        "CC1=CC(=CC(=C1O)C)C(=O)C",
        "CCN(CC)S(=O)(=O)C1=CC=C(C=C1)C(=O)OCC",
        "C[Si](C)(C)CC1=CC=CC=C1",
        "CN1C=NC2=C1C(=O)NC(=O)N2C",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]


@pytest.fixture(scope="module")
def dataset_smiles(test_data):
    return test_data["smiles"].values


@pytest.fixture(scope="module")
def dataset_targets(test_data):
    return test_data["expt"].values
