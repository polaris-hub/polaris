import pytest
from pydantic import ValidationError
from pyroaring import BitMap

from polaris.benchmark._benchmark_v2 import BenchmarkV2Specification
from polaris.benchmark._split_v2 import IndexSet, SplitV2


@pytest.fixture
def index_set():
    return IndexSet(indices=BitMap([1, 2, 3, 4, 5]))


@pytest.fixture
def valid_split():
    return SplitV2(training=IndexSet(indices=BitMap([0, 1, 2])), test=IndexSet(indices=BitMap([3, 4, 5])))


def test_index_set_creation():
    """Test IndexSet initialization and validation"""
    # Test with sequence
    index_set = IndexSet(indices=[1, 2, 3])
    assert isinstance(index_set.indices, BitMap)
    assert index_set.datapoints == 3

    # Test with BitMap
    bitmap = BitMap([1, 2, 3])
    index_set = IndexSet(indices=bitmap)
    assert isinstance(index_set.indices, BitMap)
    assert index_set.datapoints == 3


def test_split_v2_creation():
    """Test SplitV2 initialization and validation"""
    train_set = IndexSet(indices=[0, 1, 2])
    test_set = IndexSet(indices=[3, 4, 5])

    split = SplitV2(training=train_set, test=test_set)
    assert split.n_train_datapoints == 3
    assert split.n_test_datapoints == 3
    assert split.max_index == 5


def test_split_v2_overlap_validation():
    """Test that overlapping train/test sets raise an error"""
    train_set = IndexSet(indices=[0, 1, 2])
    test_set = IndexSet(indices=[2, 3, 4])  # Overlapping with train

    with pytest.raises(ValidationError):
        SplitV2(training=train_set, test=test_set)


def test_split_v2_empty_test_validation():
    """Test that empty test sets raise an error"""
    train_set = IndexSet(indices=[0, 1, 2])
    test_set = IndexSet(indices=[])

    with pytest.raises(ValidationError):
        SplitV2(training=train_set, test=test_set)


def test_split_v2_empty_train_allowed():
    """Test that empty training sets are allowed (zero-shot scenarios)"""
    train_set = IndexSet(indices=[])  # Empty training set
    test_set = IndexSet(indices=[0, 1, 2])

    # This should not raise an error
    split = SplitV2(training=train_set, test=test_set)
    assert split.n_train_datapoints == 0
    assert split.n_test_datapoints == 3
    assert split.max_index == 2


def test_split_specification_v2_mixin_creation():
    """Test SplitSpecificationV2Mixin functionality with multiple splits"""
    from polaris.benchmark._split_v2 import SplitSpecificationV2Mixin

    splits = {
        "split1": SplitV2(training=IndexSet(indices=[0, 1]), test=IndexSet(indices=[2, 3])),
        "split2": SplitV2(training=IndexSet(indices=[0, 2]), test=IndexSet(indices=[1, 3])),
    }

    # Create a mock class that inherits from the mixin
    class MockBenchmark(SplitSpecificationV2Mixin):
        pass

    mock_benchmark = MockBenchmark(splits=splits)
    assert mock_benchmark.n_splits == 2
    assert set(mock_benchmark.split_labels) == {"split1", "split2"}
    assert mock_benchmark.n_train_datapoints == {"split1": 2, "split2": 2}
    assert mock_benchmark.n_test_datapoints == {"split1": 2, "split2": 2}


def test_split_specification_v2_mixin_empty_splits_validation():
    """Test that empty splits dict raises an error"""
    from polaris.benchmark._split_v2 import SplitSpecificationV2Mixin

    class MockBenchmark(SplitSpecificationV2Mixin):
        pass

    with pytest.raises(ValidationError):
        MockBenchmark(splits={})


def test_split_specification_v2_mixin_properties(valid_split):
    """
    Test the properties of SplitSpecificationV2Mixin with a single split
    """
    from polaris.benchmark._split_v2 import SplitSpecificationV2Mixin

    class MockBenchmark(SplitSpecificationV2Mixin):
        pass

    mock_benchmark = MockBenchmark(splits={"test": valid_split})
    assert mock_benchmark.n_splits == 1
    assert mock_benchmark.split_labels == ["test"]
    assert mock_benchmark.n_train_datapoints == {"test": 3}
    assert mock_benchmark.n_test_datapoints == {"test": 3}
    assert mock_benchmark.max_index == 5


def test_benchmark_v2_specification(valid_split, test_dataset_v2, tmp_path):
    """
    Test BenchmarkV2Specification initialization and validation
    """
    # Mock minimal valid configuration
    config = {
        "dataset": test_dataset_v2,
        "splits": {"test": valid_split},
        "target_cols": ["B"],
        "input_cols": ["A"],
        "n_classes": {"B": 2},
        "name": "test_benchmark",
    }

    # Test valid configuration
    benchmark = BenchmarkV2Specification(**config)
    assert benchmark.n_splits == 1
    assert benchmark.split_labels == ["test"]
    assert benchmark.n_train_datapoints == {"test": 3}
    assert benchmark.n_test_datapoints == {"test": 3}

    # Test dict as dataset input
    config = {
        "dataset": test_dataset_v2.model_dump(),
        "splits": {"test": valid_split},
        "target_cols": ["B"],
        "input_cols": ["A"],
        "n_classes": {"B": 2},
        "name": "test_benchmark",
    }
    BenchmarkV2Specification(**config)


def test_benchmark_v2_invalid_indices(valid_split, test_dataset_v2):
    """
    Test validation of indices against dataset length
    """
    max_index = len(test_dataset_v2)
    invalid_split = SplitV2(
        training=IndexSet(indices=BitMap([0, 1])),
        test=IndexSet(indices=BitMap([max_index])),  # Invalid index
    )

    config = {
        "dataset": test_dataset_v2,
        "splits": {"test": invalid_split},
        "target_cols": ["B"],
        "input_cols": ["A"],
        "metrics": ["accuracy", "r2"],
        "main_metric": "accuracy",
        "n_classes": {"B": 2},
        "name": "test_benchmark",
    }

    with pytest.raises(ValidationError):
        BenchmarkV2Specification(**config)


def test_benchmark_v2_n_classes_validation(test_dataset_v2):
    """Test n_classes validation"""
    valid_split = SplitV2(training=IndexSet(indices=BitMap([0, 1])), test=IndexSet(indices=BitMap([2, 3])))

    config = {
        "dataset": test_dataset_v2,
        "splits": {"test": valid_split},
        "target_cols": ["B"],
        "input_cols": ["A"],
        "metrics": ["accuracy", "r2"],
        "main_metric": "accuracy",
        "n_classes": {"invalid_target": 2},  # Invalid target column
        "name": "test_benchmark",
    }

    with pytest.raises(ValidationError):
        BenchmarkV2Specification(**config)


def test_benchmark_v2_with_multiple_test_sets(test_benchmark_v2_multiple_test_sets):
    """Test BenchmarkV2 with multiple test sets functionality using fixture"""
    benchmark = test_benchmark_v2_multiple_test_sets

    # Test split properties
    assert benchmark.n_splits == 3
    assert set(benchmark.split_labels) == {"split_1", "split_2", "split_3"}
    assert benchmark.n_train_datapoints == {"split_1": 5, "split_2": 6, "split_3": 5}
    assert benchmark.n_test_datapoints == {"split_1": 3, "split_2": 2, "split_3": 4}

    # Test train-test split
    splits_data = benchmark.get_train_test_split()
    assert isinstance(splits_data, dict)
    assert len(splits_data) == 3

    # Check each split
    for split_name in ["split_1", "split_2", "split_3"]:
        train, test = splits_data[split_name]
        assert hasattr(train, "__len__")  # Train subset exists
        assert hasattr(test, "__len__")  # Test subset exists

    # Test split_items() generator
    split_items = list(benchmark.split_items())
    assert len(split_items) == 3
    labels = [label for label, _ in split_items]
    assert set(labels) == {"split_1", "split_2", "split_3"}


def test_split_specification_v2_mixin_validation_multiple_splits():
    """Test validation rules for multiple splits"""
    from polaris.benchmark._split_v2 import SplitSpecificationV2Mixin

    class MockBenchmark(SplitSpecificationV2Mixin):
        pass

    # Test empty splits should fail
    with pytest.raises(ValidationError):
        MockBenchmark(splits={})

    # Test empty individual test set should fail
    with pytest.raises(ValidationError):
        MockBenchmark(
            splits={
                "test1": SplitV2(training=IndexSet(indices=BitMap([0, 1])), test=IndexSet(indices=BitMap([])))
            }
        )

    # Test overlapping train and test sets should fail
    with pytest.raises(ValidationError):
        MockBenchmark(
            splits={
                "test1": SplitV2(
                    training=IndexSet(indices=BitMap([0, 1, 2])), test=IndexSet(indices=BitMap([2, 3, 4]))
                ),  # 2 overlaps
            }
        )

    # Test valid multiple splits should pass
    mock_benchmark = MockBenchmark(
        splits={
            "split1": SplitV2(
                training=IndexSet(indices=BitMap([0, 1, 2])), test=IndexSet(indices=BitMap([3, 4]))
            ),
            "split2": SplitV2(
                training=IndexSet(indices=BitMap([0, 2])), test=IndexSet(indices=BitMap([5, 6, 7]))
            ),
        }
    )
    assert mock_benchmark.n_splits == 2
    assert mock_benchmark.max_index == 7


def test_benchmark_v2_get_train_test_split_method(test_dataset_v2, test_org_owner):
    """Test the main get_train_test_split API with various scenarios"""

    # Test with multiple splits
    splits = {
        "fold_1": SplitV2(training=IndexSet(indices=[0, 1, 2]), test=IndexSet(indices=[3, 4])),
        "fold_2": SplitV2(training=IndexSet(indices=[0, 3, 4]), test=IndexSet(indices=[1, 2])),
        "zero_shot": SplitV2(
            training=IndexSet(indices=[]),  # Empty training set
            test=IndexSet(indices=[5, 6]),
        ),
    }

    benchmark = BenchmarkV2Specification(
        name="test-get-split-method",
        owner=test_org_owner,
        dataset=test_dataset_v2,
        splits=splits,
        target_cols=["A"],
        input_cols=["B"],
    )

    # Test the main API
    splits_result = benchmark.get_train_test_split()

    # Should return a dictionary with all splits
    assert isinstance(splits_result, dict)
    assert set(splits_result.keys()) == {"fold_1", "fold_2", "zero_shot"}

    # Test each split
    fold1_train, fold1_test = splits_result["fold_1"]
    assert len(fold1_train) == 3
    assert len(fold1_test) == 2

    fold2_train, fold2_test = splits_result["fold_2"]
    assert len(fold2_train) == 3
    assert len(fold2_test) == 2

    # Test zero-shot scenario (empty training set)
    zero_shot_train, zero_shot_test = splits_result["zero_shot"]
    assert len(zero_shot_train) == 0
    assert len(zero_shot_test) == 2

    # Test with featurization function
    def dummy_featurization(x):
        return x

    splits_with_feat = benchmark.get_train_test_split(featurization_fn=dummy_featurization)
    assert isinstance(splits_with_feat, dict)
    assert len(splits_with_feat) == 3
