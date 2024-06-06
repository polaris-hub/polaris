import numpy as np
import pandas as pd
from polaris.evaluate.utils import evaluate_benchmark

from polaris.competition import CompetitionSpecification

def test_competition_from_json(test_competition, tmpdir):
    """Test whether we can successfully save and load a competition from JSON."""
    path = test_competition.to_json(str(tmpdir))
    new_competition = CompetitionSpecification.from_json(path)
    assert new_competition == test_competition

def test_competition_evaluation(test_competition):
    """Test that multi-task competitions will be evaluated properly when when
    target labels are read as a pandas dataframe from a file."""
    data = np.random.randint(2, size=(6, 3))
    labels = pd.DataFrame(data, columns=['Column1', 'Column2', 'Column3'])
    labels_as_from_hub = {col: np.array(labels[col]) for col in labels.columns}
    predictions = {
        target_col: np.random.randint(2, size=labels.shape[0]) for target_col in labels.columns
    }

    result = evaluate_benchmark(predictions,
                                labels_as_from_hub,
                                ['Column1', 'Column2', 'Column3'],
                                test_competition.metrics)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "Test set",
        "Target label",
        "Metric",
        "Score",
    }