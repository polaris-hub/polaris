import pytest

import datamol as dm

from polaris.splitter.simpd import SIMPDSplitter


def test_simpd():
    data = dm.freesolv()
    data = data.iloc[:100]
    data["mol"] = data["smiles"].apply(dm.to_mol)
    data["active"] = data["expt"] < -8

    args = {}
    args["n_splits"] = 5

    # GA parameters
    args["pop_size"] = 500
    args["ngens"] = 2

    # SIMPD objectives and constraints
    args["simpd_descriptors"] = None
    args["target_train_frac_active"] = -1
    args["target_test_frac_active"] = -1
    args["target_delta_test_frac_active"] = None  # [0.11, 0.30] or None
    args["target_GF_delta_window"] = (10, 30)
    args["target_G_val"] = 70
    args["max_population_cluster_entropy"] = 0.9
    args["pareto_weight_GF_delta"] = 10
    args["pareto_weight_G"] = 5

    # Misc
    args["num_threads"] = 1
    args["random_seed"] = 15
    args["verbose"] = True
    args["verbose_pymoo"] = True
    args["progress"] = True
    args["progress_leave"] = True

    splitter = SIMPDSplitter(**args)

    X = data["mol"].to_numpy()
    y = data["active"].to_numpy()

    results = splitter.fit(X=X, y=y)

    for train_ind, test_ind in splitter.split(X):
        assert len(train_ind) + len(test_ind) == len(data)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > len(test_ind)
        assert len(train_ind) > 0 and len(test_ind) > 0


def test_simpd_not_binary_fails():
    data = dm.freesolv()
    data = data.iloc[:100]
    data["mol"] = data["smiles"].apply(dm.to_mol)

    splitter = SIMPDSplitter()
    X = data["mol"].to_numpy()
    y = data["expt"].to_numpy()

    with pytest.raises(ValueError):
        splitter.fit(X=X, y=y)
