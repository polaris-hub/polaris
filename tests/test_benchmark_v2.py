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
        "metrics": ["accuracy", "r2"],
        "main_metric": "accuracy",
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
        "metrics": ["accuracy", "r2"],
        "main_metric": "accuracy",
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
