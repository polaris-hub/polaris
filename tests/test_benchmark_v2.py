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


def test_index_set_serialization(index_set):
    """
    Test IndexSet serialization and deserialization
    """
    serialized = index_set.serialize()
    deserialized = IndexSet.deserialize(serialized)
    assert deserialized.indices == index_set.indices
    assert deserialized.datapoints == index_set.datapoints
    assert deserialized.md5_checksum == index_set.md5_checksum


def test_split_v2_serialized_initialization(valid_split):
    """
    Test that SplitV2 can be initialized from serialized IndexSets
    """
    test = valid_split.test.serialize()
    training = valid_split.training.serialize()

    split = SplitV2(training=training, test=test)
    assert split.training.indices == valid_split.training.indices
    assert split.test.indices == valid_split.test.indices


def test_split_v2_validation():
    """Test SplitV2 validation rules"""
    # Test empty test set
    with pytest.raises(ValidationError):
        SplitV2(training=IndexSet(indices=BitMap([1, 2])), test=IndexSet(indices=BitMap([])))

    # Test overlapping sets
    with pytest.raises(ValidationError):
        SplitV2(training=IndexSet(indices=BitMap([1, 2])), test=IndexSet(indices=BitMap([2, 3])))

    # Valid split should not raise
    SplitV2(training=IndexSet(indices=BitMap([1, 2])), test=IndexSet(indices=BitMap([3, 4])))


def test_split_v2_properties(valid_split):
    """
    Test the properties of SplitV2 reflect the underlying index sets
    """
    assert valid_split.n_train_datapoints == valid_split.training.datapoints
    assert valid_split.n_test_datapoints == {"test": valid_split.test.datapoints}
    assert valid_split.n_test_sets == 1
    assert valid_split.max_index == 5


def test_benchmark_v2_specification(valid_split, test_dataset_v2, tmp_path):
    """
    Test BenchmarkV2Specification initialization and validation
    """
    # Mock minimal valid configuration
    config = {
        "dataset": test_dataset_v2,
        "split": valid_split,
        "target_cols": ["B"],
        "input_cols": ["A"],
        "n_classes": {"B": 2},
        "name": "test_benchmark",
    }

    # Test valid configuration
    benchmark = BenchmarkV2Specification(**config)
    assert benchmark.n_train_datapoints == 3
    assert benchmark.n_test_datapoints == {"test": 3}
    assert benchmark.test_set_labels == ["test"]

    # Test dict as dataset input
    config = {
        "dataset": test_dataset_v2.model_dump(),
        "split": valid_split,
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
    config = {
        "dataset": test_dataset_v2,
        "split": SplitV2(
            training=IndexSet(indices=BitMap([0, 1])),
            test=IndexSet(indices=BitMap([max_index, max_index + 1])),  # Invalid indices
        ),
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
    config = {
        "dataset": test_dataset_v2,
        "split": SplitV2(training=IndexSet(indices=BitMap([0, 1])), test=IndexSet(indices=BitMap([2, 3]))),
        "target_cols": ["B"],
        "input_cols": ["A"],
        "metrics": ["accuracy", "r2"],
        "main_metric": "accuracy",
        "n_classes": {"invalid_target": 2},  # Invalid target column
        "name": "test_benchmark",
    }

    with pytest.raises(ValidationError):
        BenchmarkV2Specification(**config)


def test_benchmark_v2_multiple_test_sets(test_dataset_v2, test_org_owner):
    """Test BenchmarkV2 with multiple test sets functionality"""
    
    # Create split with multiple test sets
    split = SplitV2(
        training=IndexSet(indices=BitMap([0, 1, 2, 3, 4])),
        test_sets={
            "test_set_1": IndexSet(indices=BitMap([5, 6, 7])),
            "test_set_2": IndexSet(indices=BitMap([8, 9])),
            "test_set_3": IndexSet(indices=BitMap([10, 11, 12, 13])),
        }
    )
    
    benchmark = BenchmarkV2Specification(
        name="test-benchmark-multiple-test-sets",
        owner=test_org_owner,
        dataset=test_dataset_v2,
        split=split,
        target_cols=["A"],
        input_cols=["B"],
        metrics=["mean_absolute_error"],
        main_metric="mean_absolute_error",
    )
    
    # Test split properties
    assert benchmark.n_train_datapoints == 5
    assert benchmark.n_test_sets == 3
    assert benchmark.n_test_datapoints == {"test_set_1": 3, "test_set_2": 2, "test_set_3": 4}
    assert set(benchmark.test_set_labels) == {"test_set_1", "test_set_2", "test_set_3"}
    assert benchmark.test_set_sizes == {"test_set_1": 3, "test_set_2": 2, "test_set_3": 4}
    
    # Test train-test split
    train, test = benchmark.get_train_test_split()
    assert len(train) == 5
    assert isinstance(test, dict)
    assert len(test) == 3
    assert len(test["test_set_1"]) == 3
    assert len(test["test_set_2"]) == 2
    assert len(test["test_set_3"]) == 4
    
    # Test backward compatibility - accessing .test property should work
    assert split.test.datapoints > 0  # Should return first test set or combined
    
    # Test test_items() generator
    test_items = list(split.test_items())
    assert len(test_items) == 3
    labels = [label for label, _ in test_items]
    assert set(labels) == {"test_set_1", "test_set_2", "test_set_3"}


def test_split_v2_backward_compatibility():
    """Test that SplitV2 maintains backward compatibility with single test field"""
    
    # Test creating split with old 'test' field format
    split_data = {
        "training": IndexSet(indices=BitMap([0, 1, 2])),
        "test": IndexSet(indices=BitMap([3, 4, 5]))
    }
    
    split = SplitV2(**split_data)
    
    # Should convert single test to test_sets format
    assert split.n_test_sets == 1
    assert "test" in split.test_sets
    assert split.test_sets["test"].datapoints == 3
    assert split.test.datapoints == 3  # Backward compatibility property
    
    # Test with serialized data
    test_serialized = IndexSet(indices=BitMap([6, 7, 8])).serialize()
    training_serialized = IndexSet(indices=BitMap([0, 1, 2])).serialize()
    
    split2 = SplitV2(training=training_serialized, test=test_serialized)
    assert split2.n_test_sets == 1
    assert split2.test.datapoints == 3


def test_split_v2_validation_multiple_test_sets():
    """Test validation rules for multiple test sets"""
    
    # Test empty test sets should fail
    with pytest.raises(ValidationError):
        SplitV2(
            training=IndexSet(indices=BitMap([0, 1])),
            test_sets={}
        )
    
    # Test empty individual test set should fail
    with pytest.raises(ValidationError):
        SplitV2(
            training=IndexSet(indices=BitMap([0, 1])),
            test_sets={"test1": IndexSet(indices=BitMap([]))}
        )
    
    # Test overlapping train and test sets should fail
    with pytest.raises(ValidationError):
        SplitV2(
            training=IndexSet(indices=BitMap([0, 1, 2])),
            test_sets={"test1": IndexSet(indices=BitMap([2, 3, 4]))}  # 2 overlaps
        )
    
    # Test valid multiple test sets should pass
    split = SplitV2(
        training=IndexSet(indices=BitMap([0, 1, 2])),
        test_sets={
            "test1": IndexSet(indices=BitMap([3, 4])),
            "test2": IndexSet(indices=BitMap([5, 6, 7])),
        }
    )
    assert split.n_test_sets == 2
    assert split.max_index == 7
