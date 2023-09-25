import pytest

from polaris.dataset import Subset
from polaris.utils.errors import TestAccessError


def test_consistency_across_access_methods(test_dataset):
    """Using the various endpoints of the Subset API should not lead to the same data."""
    indices = list(range(5))
    task = Subset(test_dataset, indices, "smiles", "expt")

    # Ground truth
    expected_smiles = test_dataset.table.loc[indices, "smiles"]
    expected_targets = test_dataset.table.loc[indices, "expt"]

    # Indexing
    assert ([task[i][0] for i in range(5)] == expected_smiles).all()
    assert ([task[i][1] for i in range(5)] == expected_targets).all()

    # Iterator
    assert (list(smi for smi, y in task) == expected_smiles).all()
    assert (list(y for smi, y in task) == expected_targets).all()

    # Property
    assert (task.inputs == expected_smiles).all()
    assert (task.targets == expected_targets).all()


def test_access_to_test_set(test_single_task_benchmark):
    """A user should not have access to the test set targets."""

    train, test = test_single_task_benchmark.get_train_test_split()
    assert test._hide_targets
    assert not train._hide_targets

    with pytest.raises(TestAccessError):
        test.as_array("y")
    with pytest.raises(TestAccessError):
        test.targets

    # Check if iterable style access returns just the SMILES
    for x in test:
        assert isinstance(x, str)
    for i in range(len(test)):
        assert isinstance(test[i], str)

    # For the train set it should work
    assert all(isinstance(y, float) for x, y in train)
    assert all(isinstance(train[i][1], float) for i in range(len(train)))
