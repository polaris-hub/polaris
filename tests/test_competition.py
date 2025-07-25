from itertools import chain

import pytest
from pydantic import ValidationError

from polaris.competition import CompetitionSpecification
from polaris.utils.types import TaskType


def test_competition_split_verification(test_competition):
    """Verifies that the split validation works as expected."""

    obj = test_competition
    cls = CompetitionSpecification

    # By using the fixture as a default, we know it doesn't always fail
    default_kwargs = {
        "target_cols": obj.target_cols,
        "input_cols": obj.input_cols,
        "name": obj.name,
        "zarr_root_path": obj.zarr_root_path,
        "readme": obj.readme,
        "start_time": obj.start_time,
        "end_time": obj.end_time,
        "n_test_sets": obj.n_test_sets,
        "n_test_datapoints": obj.n_test_datapoints,
        "n_classes": obj.n_classes,
    }

    train_split = obj.split[0]
    test_split = obj.split[1]

    # One or more empty test partitions
    with pytest.raises(ValidationError):
        cls(split=(train_split,), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, []), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, {"test": []}), **default_kwargs)
    # Non-exclusive partitions
    with pytest.raises(ValidationError):
        cls(split=(train_split, test_split["test"] + train_split[:1]), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, {"test1": test_split, "test2": train_split[:1]}), **default_kwargs)
    # Invalid indices
    with pytest.raises(ValidationError):
        cls(split=(train_split + [len(obj)], test_split), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split + [-1], test_split), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, test_split["test"] + [len(obj)]), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, test_split["test"] + [-1]), **default_kwargs)
    # Duplicate indices
    with pytest.raises(ValidationError):
        cls(split=(train_split + train_split[:1], test_split), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(split=(train_split, test_split["test"] + test_split["test"][:1]), **default_kwargs)
    with pytest.raises(ValidationError):
        cls(
            split=(train_split, {"test1": test_split, "test2": test_split["test"] + test_split["test"][:1]}),
            **default_kwargs,
        )

    # It should _not_ fail with duplicate indices across test partitions
    cls(split=(train_split, {"test1": test_split["test"], "test2": test_split["test"]}), **default_kwargs)
    # It should _not_ fail with missing indices
    cls(split=(train_split[:-1], test_split), **default_kwargs)
    # It should _not_ fail with an empty train set
    competition = cls(split=([], test_split), **default_kwargs)
    train, _ = competition.get_train_test_split()
    assert len(train) == 0


def test_competition_metric_deserialization(test_competition):
    """Tests that passing metrics as a list of strings or dictionaries works as expected"""
    m = test_competition.model_dump()

    # Should work with strings
    m["metrics"] = ["mean_absolute_error", "accuracy"]
    m["main_metric"] = "accuracy"
    CompetitionSpecification(**m)

    # Should work with dictionaries
    m["metrics"] = [
        {"label": "mean_absolute_error", "config": {"group_by": "CLASS_expt"}},
        {"label": "accuracy"},
    ]
    CompetitionSpecification(**m)


def test_competition_train_test_split(test_competition):
    """Tests that the competition's train/test split can be retrieved through a CompetitionSpecification instance"""

    train, test = test_competition.get_train_test_split()

    train_split = test_competition.split[0]
    test_sets = test_competition.split[1]
    test_split = set(chain.from_iterable(test_sets.values()))

    assert len(train) == len(train_split)
    assert len(test) == len(test_split)


def test_competition_computed_fields(test_competition):
    default_test_set_name = "test"
    assert test_competition.task_type == TaskType.SINGLE_TASK.value
    assert test_competition.test_set_labels == [default_test_set_name]
    assert test_competition.test_set_sizes == {default_test_set_name: 10}


def test_competition_interface(test_competition):
    """Tests that the CompetitionSpecification class doesn't accidentally inherit the evaluate method from the benchmark class"""
    with pytest.raises(AttributeError):
        test_competition.evaluate()
