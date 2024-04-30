import datamol as dm
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from polaris.evaluate import BenchmarkResults


def test_single_task_benchmark_loop(test_single_task_benchmark):
    """Tests the integrated API for a single-task benchmark."""
    train, test = test_single_task_benchmark.get_train_test_split()

    model = RandomForestRegressor()
    smiles, y = train.as_array("xy")
    x = np.array([dm.to_fp(dm.to_mol(smi)) for smi in smiles])
    model.fit(X=x, y=y)

    x = np.array([dm.to_fp(dm.to_mol(smi)) for smi in test.inputs])
    y_pred = model.predict(x)

    scores = test_single_task_benchmark.evaluate(y_pred)
    assert isinstance(scores, BenchmarkResults)


def test_single_task_benchmark_loop_with_multiple_test_sets(test_single_task_benchmark_multiple_test_sets):
    """Tests the integrated API for a single-task benchmark with multiple test sets."""
    train, test = test_single_task_benchmark_multiple_test_sets.get_train_test_split()

    smiles, y = train.as_array("xy")

    x_train = np.array([dm.to_fp(dm.to_mol(smi)) for smi in smiles])

    model = RandomForestRegressor()
    model.fit(X=x_train, y=y)

    y_pred = {}
    for k, test_subset in test.items():
        x_test = np.array([dm.to_fp(dm.to_mol(smi)) for smi in test_subset.inputs])
        y_pred[k] = model.predict(x_test)

    scores = test_single_task_benchmark_multiple_test_sets.evaluate(y_pred)
    assert isinstance(scores, BenchmarkResults)


def test_single_task_benchmark_clf_loop_with_multiple_test_sets(
    test_single_task_benchmark_clf_multiple_test_sets,
):
    """Tests the integrated API for a single-task benchmark for classification probabilities with multiple test sets."""
    train, test = test_single_task_benchmark_clf_multiple_test_sets.get_train_test_split()

    smiles, y = train.as_array("xy")

    x_train = np.array([dm.to_fp(dm.to_mol(smi)) for smi in smiles])

    model = RandomForestClassifier()
    model.fit(X=x_train, y=y)

    y_prob = {}
    y_pred = {}
    for k, test_subset in test.items():
        print(k, test_subset)
        x_test = np.array([dm.to_fp(dm.to_mol(smi)) for smi in test_subset.inputs])
        y_prob[k] = model.predict_proba(x_test)[:, :1]  # for binary classification
        y_pred[k] = model.predict(x_test)

    scores = test_single_task_benchmark_clf_multiple_test_sets.evaluate(y_prob=y_prob, y_pred=y_pred)
    assert isinstance(scores, BenchmarkResults)


def test_multi_task_benchmark_loop(test_multi_task_benchmark):
    """Tests the integrated API for a multi-task benchmark."""
    train, test = test_multi_task_benchmark.get_train_test_split()

    smiles, multi_y = train.as_array("xy")
    x_train = np.array([dm.to_fp(dm.to_mol(smi)) for smi in smiles])
    x_test = np.array([dm.to_fp(dm.to_mol(smi)) for smi in test.inputs])

    y_pred = {}
    for k, y in multi_y.items():
        model = RandomForestRegressor()

        mask = ~np.isnan(y)
        model.fit(X=x_train[mask], y=y[mask])
        y_pred[k] = model.predict(x_test)

    scores = test_multi_task_benchmark.evaluate(y_pred)
    assert isinstance(scores, BenchmarkResults)
