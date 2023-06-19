from polaris.data import Subset


def test_single_task_indexing(test_dataset):
    target_col = "expt"
    input_col = "smiles"

    task = Subset(test_dataset, list(range(5)), input_col, target_col)

    assert len(task) == 5
    for i in range(5):
        assert test_dataset.table.loc[i, input_col] == task[i][0]
        assert test_dataset.table.loc[i, target_col] == task[i][1]


def test_multi_task_indexing(test_dataset):
    target_cols = ["expt", "calc"]
    input_col = "smiles"

    task = Subset(test_dataset, [(i, [0, 1]) for i in range(5)], input_col, target_cols)

    assert len(task) == 5
    for i in range(5):
        assert test_dataset.table.loc[i, input_col] == task[i][0]
        assert (test_dataset.table.loc[i, target_cols].values == task[i][1]).all()

    # With sparsity (indices have changed)
    task = Subset(test_dataset, [(i, [0]) for i in range(5)], input_col, target_cols)
    assert len(task) == 5
    for i in range(5):
        assert test_dataset.table.loc[i, input_col] == task[i][0]
        assert test_dataset.table.loc[i, target_cols[0]] == task[i][1][0]


def test_consistency_across_access_methods(test_dataset):
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
