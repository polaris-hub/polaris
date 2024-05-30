import numpy as np

from polaris.competition import CompetitionSpecification

test = [-3.84, -9.73, -2.49, -4.13, -4.02, -2.1 , -4.59,  2.3 , -6.09, -7.07]
predictions = test + np.random.uniform(0, 3, size=len(test))

def test_competition_from_json(test_competition, tmpdir):
    """Test whether we can successfully save and load a competition from JSON."""
    path = test_competition.to_json(str(tmpdir))
    new_competition = CompetitionSpecification.from_json(path)
    assert new_competition == test_competition
