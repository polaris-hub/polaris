import numpy as np
import pandas as pd

from polaris.competition import CompetitionSpecification
from polaris.evaluate.utils import evaluate_benchmark


def test_competition_from_json(test_competition, tmp_path):
    """Test whether we can successfully save and load a competition from JSON."""
    path = test_competition.to_json(str(tmp_path))
    new_competition = CompetitionSpecification.from_json(path)
    assert new_competition == test_competition


def test_multi_col_competition_evaluation(test_competition):
    """Test that multi-column competitions will be evaluated properly when when
    target labels are read as a pandas dataframe from a file."""
    data = np.random.randint(2, size=(10, 3))
    labels = pd.DataFrame(data, columns=["Column1", "Column2", "Column3"])
    labels_as_from_hub = {col: np.array(labels[col]) for col in labels.columns}
    predictions = {target_col: np.random.randint(2, size=labels.shape[0]) for target_col in labels.columns}

    result = evaluate_benchmark(
        target_cols=["Column1", "Column2", "Column3"],
        test_set_labels=["test"],
        test_set_sizes=test_competition.test_set_sizes,
        metrics=test_competition.metrics,
        y_true=labels_as_from_hub,
        y_pred=predictions,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "Test set",
        "Target label",
        "Metric",
        "Score",
    }


def test_single_col_competition_evaluation(test_competition):
    """Test that multi-column competitions will be evaluated properly when when
    target labels are read as a pandas dataframe from a file."""
    y_true = np.array(
        [
            1.15588236,
            1.56414507,
            1.04828639,
            0.98362629,
            1.22613572,
            2.56594576,
            0.67568671,
            0.86099644,
            0.67568671,
            2.28213589,
        ]
    )

    y_pred = y_true + np.random.uniform(0, 3, size=len(y_true))

    result = evaluate_benchmark(
        target_cols=["LOG HLM_CLint (mL/min/kg)"],
        test_set_labels=["test"],
        test_set_sizes=test_competition.test_set_sizes,
        metrics=test_competition.metrics,
        y_true=y_true,
        y_pred=y_pred,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "Test set",
        "Target label",
        "Metric",
        "Score",
    }
