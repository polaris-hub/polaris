import pandas as pd
import numpy as np

from polaris.competition import CompetitionSpecification

def test_competition_from_json(test_competition, tmpdir):
    """Test whether we can successfully save and load a competition from JSON."""
    path = test_competition.to_json(str(tmpdir))
    new_competition = CompetitionSpecification.from_json(path)
    assert new_competition == test_competition

def test_competition_evaluation(test_competition):
    """Test whether we can successfully evaluate a competition."""
    competition = test_competition
    test = [-3.84, -9.73, -2.49, -4.13, -4.02, -2.1 , -4.59,  2.3 , -6.09, -7.07]
    predictions = test + np.random.uniform(0, 3, size=len(test))
    result = competition._hub_evaluate(predictions, test)
    assert isinstance(result.results, pd.DataFrame)
    assert set(result.results.columns) == {
        "Test set",
        "Target label",
        "Metric",
        "Score",
    }
    for metric in competition.metrics:
        assert metric in result.results.Metric.tolist()