import numpy as np
import pandas as pd
import pytest

from pydantic import ValidationError
from polaris.evaluate.utils import evaluate_benchmark, normalize_predictions_type
from polaris.competition import CompetitionSpecification
from polaris.dataset import Dataset
from polaris.utils.types import HubOwner


def test_competition_from_json(test_competition, tmpdir):
    """Test whether we can successfully save and load a competition from JSON."""
    path = test_competition.to_json(str(tmpdir))
    new_competition = CompetitionSpecification.from_json(path)
    assert new_competition == test_competition


def test_multi_col_competition_evaluation(test_competition):
    """Test that multi-column competitions will be evaluated properly when when
    target labels are read as a pandas dataframe from a file."""
    data = np.random.randint(2, size=(6, 3))
    labels = pd.DataFrame(data, columns=["Column1", "Column2", "Column3"])
    labels_as_from_hub = {col: np.array(labels[col]) for col in labels.columns}
    predictions = {target_col: np.random.randint(2, size=labels.shape[0]) for target_col in labels.columns}

    result = evaluate_benchmark(
        ["Column1", "Column2", "Column3"], test_competition.metrics, labels_as_from_hub, y_pred=predictions
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
    data = np.array(
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
            1.06617679,
            1.05709529,
            0.67568671,
            0.67568671,
            0.67568671,
        ]
    )
    labels = {"LOG HLM_CLint (mL/min/kg)": data}
    predictions = data + np.random.uniform(0, 3, size=len(data))

    result = evaluate_benchmark(
        ["LOG HLM_CLint (mL/min/kg)"], test_competition.metrics, labels, y_pred=predictions
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "Test set",
        "Target label",
        "Metric",
        "Score",
    }


def test_normalize_predictions_type():
    "Single column, single test set"
    assert {"test": {"col1": [1, 2, 3]}} == normalize_predictions_type([1, 2, 3], ["col1"])
    assert {"test": {"col1": [1, 2, 3]}} == normalize_predictions_type({"col1": [1, 2, 3]}, ["col1"])
    assert {"test": {"col1": [1, 2, 3]}} == normalize_predictions_type(
        {"test": {"col1": [1, 2, 3]}}, ["col1"]
    )

    "Multi-column, single test set"
    assert {"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}} == normalize_predictions_type(
        {"col1": [1, 2, 3], "col2": [4, 5, 6]}, ["col1", "col2"]
    )

    assert {"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}} == normalize_predictions_type(
        {"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}}, ["col1", "col2"]
    )

    "Single column, multi-test set"
    assert {"test1": {"col1": [1, 2, 3]}, "test2": {"col1": [4, 5, 6]}} == normalize_predictions_type(
        {"test1": {"col1": [1, 2, 3]}, "test2": {"col1": [4, 5, 6]}}, ["col1"]
    )

    "Multi-column, multi-test set"
    assert {
        "test1": {"col1": [1, 2, 3], "col2": [4, 5, 6]},
        "test2": {"col1": [7, 8, 9], "col2": [10, 11, 12]},
    } == normalize_predictions_type(
        {"test1": {"col1": [1, 2, 3], "col2": [4, 5, 6]}, "test2": {"col1": [7, 8, 9], "col2": [10, 11, 12]}},
        ["col1", "col2"],
    )


def test_invalid_competition_creation():
    data = {"col a": [1, 2, None], "col b": [4, None, 6], "col c": [7, 8, None]}

    df = pd.DataFrame(data, index=range(3))
    dataset = Dataset(table=df, name="test-dataset")

    # Check that creating a competition where there is at least one value per test row works
    CompetitionSpecification(
        name="test-competition",
        owner=HubOwner(organizationId="test-org", slug="test-org"),
        dataset=dataset,
        metrics=["mean_absolute_error", "mean_squared_error"],
        split=([0, 1], [2]),
        target_cols=["col b"],
        input_cols=["col a", "col c"],
    )

    CompetitionSpecification(
        name="test-competition",
        owner=HubOwner(organizationId="test-org", slug="test-org"),
        dataset=dataset,
        metrics=["mean_absolute_error", "mean_squared_error"],
        split=([0, 1], [2]),
        target_cols=["col a", "col b"],
        input_cols=["col c"],
    )

    with pytest.raises(ValidationError) as ex_info:
        CompetitionSpecification(
            name="test-competition",
            owner=HubOwner(organizationId="test-org", slug="test-org"),
            dataset=dataset,
            metrics=["mean_absolute_error", "mean_squared_error"],
            split=([0, 1], [2]),
            target_cols=["col a"],
            input_cols=["col b", "col c"],
        )

    assert ex_info.match("All rows of the test set must have at least one value")

    with pytest.raises(ValidationError) as ex_info_2:
        CompetitionSpecification(
            name="test-competition",
            owner=HubOwner(organizationId="test-org", slug="test-org"),
            dataset=dataset,
            metrics=["mean_absolute_error", "mean_squared_error"],
            split=([0, 1], [2]),
            target_cols=["col a", "col c"],
            input_cols=["col b"],
        )

    assert ex_info_2.match("All rows of the test set must have at least one value")
