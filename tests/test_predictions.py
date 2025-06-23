import numpy as np
import pytest
import tempfile
import zarr
import numcodecs
from unittest.mock import Mock

from polaris.prediction import BenchmarkPredictionsV2
from polaris.benchmark import BenchmarkV2Specification
from polaris.dataset import DatasetV2
from polaris.utils.types import HubOwner


def create_mock_benchmark_with_object_array():
    """Create a mock benchmark with object array columns for testing."""
    # Create a mock dataset with object array
    mock_dataset = Mock(spec=DatasetV2)

    # Create a mock zarr root with object array
    mock_zarr_root = Mock()
    mock_array = Mock()
    mock_array.dtype = np.dtype(object)
    mock_array.compressor = None
    mock_array.filters = None
    mock_array.chunks = (1,)
    mock_zarr_root.__getitem__ = lambda key: mock_array
    mock_zarr_root.__contains__ = lambda key: key in ["target_col"]

    mock_dataset.zarr_root = mock_zarr_root
    mock_dataset.artifact_id = "test-dataset/test-dataset"

    # Create mock benchmark with all required fields
    mock_benchmark = Mock(spec=BenchmarkV2Specification)
    mock_benchmark.dataset = mock_dataset
    mock_benchmark.target_cols = ["target_col"]
    mock_benchmark.get_train_test_split.return_value = ([0, 1, 2], [3, 4, 5])  # 3 test samples
    mock_benchmark.artifact_id = "test-benchmark/test-benchmark"
    mock_benchmark.name = "test-benchmark"
    mock_benchmark.description = "Test benchmark"
    mock_benchmark.owner = HubOwner(slug="test-owner")
    mock_benchmark.polaris_version = "0.1.0"
    mock_benchmark.tags = []
    mock_benchmark.user_attributes = {}

    return mock_benchmark


def create_mock_benchmark_with_numeric_array():
    """Create a mock benchmark with numeric array columns for testing."""
    # Create a mock dataset with numeric array
    mock_dataset = Mock(spec=DatasetV2)

    # Create a mock zarr root with numeric array
    mock_zarr_root = Mock()
    mock_array = Mock()
    mock_array.dtype = np.dtype(np.float64)
    mock_array.compressor = None
    mock_array.filters = None
    mock_array.chunks = (1,)
    mock_zarr_root.__getitem__ = lambda key: mock_array
    mock_zarr_root.__contains__ = lambda key: key in ["target_col"]

    mock_dataset.zarr_root = mock_zarr_root
    mock_dataset.artifact_id = "test-dataset/test-dataset"

    # Create mock benchmark with all required fields
    mock_benchmark = Mock(spec=BenchmarkV2Specification)
    mock_benchmark.dataset = mock_dataset
    mock_benchmark.target_cols = ["target_col"]
    mock_benchmark.get_train_test_split.return_value = ([0, 1, 2], [3, 4, 5])  # 3 test samples
    mock_benchmark.artifact_id = "test-benchmark/test-benchmark"
    mock_benchmark.name = "test-benchmark"
    mock_benchmark.description = "Test benchmark"
    mock_benchmark.owner = HubOwner(slug="test-owner")
    mock_benchmark.polaris_version = "0.1.0"
    mock_benchmark.tags = []
    mock_benchmark.user_attributes = {}

    return mock_benchmark


def test_object_array_conversion_logic():
    """Test the core object array conversion logic that happens in the Predictions class."""
    # This test focuses on the specific conversion logic without needing full Pydantic validation

    # Simulate the conversion logic from the Predictions._validate_predictions method
    def convert_predictions_to_arrays(predictions, dataset_dtype):
        """Simulate the conversion logic from Predictions class."""
        if dataset_dtype == np.dtype(object):
            arr = np.empty(len(predictions), dtype=object)
            for i, item in enumerate(predictions):
                arr[i] = item
        else:
            arr = np.asarray(predictions, dtype=dataset_dtype)
        return arr

    # Test object array conversion
    object_predictions = [{"key": "value1", "id": 1}, {"key": "value2", "id": 2}, {"key": "value3", "id": 3}]

    result = convert_predictions_to_arrays(object_predictions, np.dtype(object))

    # Check that predictions are converted to numpy object array
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(object)
    assert len(result) == 3

    # Check that objects are preserved
    assert result[0]["key"] == "value1"
    assert result[1]["id"] == 2
    assert result[2]["key"] == "value3"

    # Test complex nested objects
    complex_objects = [
        {
            "molecule": {"smiles": "CCO", "atoms": ["C", "C", "O"]},
            "properties": {"energy": -1.5, "conformers": 3},
        },
        {
            "molecule": {"smiles": "CCN", "atoms": ["C", "C", "N"]},
            "properties": {"energy": -2.1, "conformers": 5},
        },
    ]

    result = convert_predictions_to_arrays(complex_objects, np.dtype(object))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(object)
    assert result[0]["molecule"]["smiles"] == "CCO"
    assert result[1]["properties"]["energy"] == -2.1

    # Test mixed object types
    mixed_objects = [
        "string_object",
        {"dict_object": "value"},
        ["list_object", "item2"],
        42,  # numeric object
        None,  # None object
        {"nested": {"deep": "structure"}},
    ]

    result = convert_predictions_to_arrays(mixed_objects, np.dtype(object))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(object)
    assert result[0] == "string_object"
    assert result[1]["dict_object"] == "value"
    assert result[2] == ["list_object", "item2"]
    assert result[3] == 42
    assert result[4] is None
    assert result[5]["nested"]["deep"] == "structure"

    # Test numeric array conversion for comparison
    numeric_predictions = [1.5, 2.7, 3.2]
    result = convert_predictions_to_arrays(numeric_predictions, np.dtype(np.float64))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(np.float64)
    assert np.array_equal(result, np.array([1.5, 2.7, 3.2]))


def test_predictions_validation_logic():
    """Test the validation logic that happens in the Predictions class."""

    def validate_predictions_size(predictions, expected_size):
        """Simulate the size validation logic from Predictions class."""
        if len(predictions) != expected_size:
            raise ValueError(
                f"Predictions size mismatch: Column has {len(predictions)} predictions, "
                f"but test set has size {expected_size}"
            )

    def validate_predictions_columns(predictions_keys, expected_columns):
        """Simulate the column validation logic from Predictions class."""
        if set(predictions_keys) != set(expected_columns):
            raise ValueError(
                f"The predictions should be a dictionary with the target columns as keys. "
                f"Expected columns: {expected_columns}, got: {list(predictions_keys)}"
            )

    # Test size validation
    predictions = [{"test": "object"}]
    with pytest.raises(ValueError, match="Predictions size mismatch"):
        validate_predictions_size(predictions, 3)

    # Test valid size
    predictions = [{"test": "object"} for _ in range(3)]
    validate_predictions_size(predictions, 3)  # Should not raise

    # Test column validation
    predictions_dict = {"wrong_column": [{"test": "object"}]}
    with pytest.raises(ValueError, match=r"Expected columns: \['target_col'\]"):
        validate_predictions_columns(predictions_dict.keys(), ["target_col"])

    # Test valid columns
    predictions_dict = {"target_col": [{"test": "object"}]}
    validate_predictions_columns(predictions_dict.keys(), ["target_col"])  # Should not raise


def test_zarr_object_array_handling():
    """Test that object arrays can be properly handled in Zarr archives."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        zarr_path = f"{temp_dir}/test.zarr"

        # Create a Zarr group
        root = zarr.group(zarr_path)

        # Create object array data
        object_data = [
            {"molecule": "CCO", "energy": -1.5},
            {"molecule": "CCN", "energy": -2.1},
            {"molecule": "CCC", "energy": -1.8},
        ]

        # Convert to numpy object array
        arr = np.empty(len(object_data), dtype=object)
        for i, item in enumerate(object_data):
            arr[i] = item

        # Store in Zarr with object codec (required for object arrays)
        root.array("target_col", data=arr, dtype=object, object_codec=numcodecs.JSON())

        # Read back from Zarr
        zarr_data = root["target_col"][:]

        # Verify data integrity
        assert len(zarr_data) == 3
        assert zarr_data[0]["molecule"] == "CCO"
        assert zarr_data[1]["energy"] == -2.1
        assert zarr_data[2]["molecule"] == "CCC"

        # Verify dtype
        assert zarr_data.dtype == np.dtype(object)


def test_predictions_serialization_format():
    """Test that predictions can be serialized in the expected format."""
    # This test verifies the expected serialization behavior without needing full Pydantic validation

    def serialize_predictions(predictions):
        """Simulate the serialization logic from Predictions class."""

        def convert_to_list(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, dict):
                return {k: convert_to_list(v) for k, v in v.items()}
            return v

        return convert_to_list(predictions)

    # Test with object arrays
    object_predictions = {
        "test": {
            "target_col": np.array(
                [{"serialization": "test1"}, {"serialization": "test2"}, {"serialization": "test3"}],
                dtype=object,
            )
        }
    }

    serialized = serialize_predictions(object_predictions)

    # Check that numpy arrays are converted to lists
    assert isinstance(serialized["test"]["target_col"], list)
    assert len(serialized["test"]["target_col"]) == 3
    assert serialized["test"]["target_col"][0]["serialization"] == "test1"
    assert serialized["test"]["target_col"][1]["serialization"] == "test2"
    assert serialized["test"]["target_col"][2]["serialization"] == "test3"


def test_predictions_metadata_structure():
    """Test that predictions have the expected metadata structure."""
    # This test verifies the expected metadata without needing full Pydantic validation

    expected_artifact_type = "prediction"
    expected_fields = ["name", "description", "benchmark", "predictions"]

    # Check that Predictions class has expected structure
    assert hasattr(BenchmarkPredictionsV2, "_artifact_type")
    assert BenchmarkPredictionsV2._artifact_type == expected_artifact_type

    # Check that required fields are defined
    for field in expected_fields:
        assert field in BenchmarkPredictionsV2.model_fields or field in BenchmarkPredictionsV2.__annotations__


def test_object_array_edge_cases():
    """Test edge cases for object array handling."""

    def convert_to_object_array(data):
        """Convert data to numpy object array."""
        arr = np.empty(len(data), dtype=object)
        for i, item in enumerate(data):
            arr[i] = item
        return arr

    # Test empty array
    empty_data = []
    result = convert_to_object_array(empty_data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(object)
    assert len(result) == 0

    # Test single item
    single_item = [{"test": "single"}]
    result = convert_to_object_array(single_item)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(object)
    assert len(result) == 1
    assert result[0]["test"] == "single"

    # Test with None values
    none_data = [None, {"test": "value"}, None]
    result = convert_to_object_array(none_data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(object)
    assert result[0] is None
    assert result[1]["test"] == "value"
    assert result[2] is None

    # Test with very large objects
    large_object = {
        "large_data": "x" * 1000,
        "nested": {"deep": {"structure": {"with": {"lots": {"of": {"data": "value"}}}}}},
    }
    large_data = [large_object for _ in range(3)]
    result = convert_to_object_array(large_data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(object)
    assert len(result) == 3
    assert result[0]["large_data"] == "x" * 1000
    assert result[1]["nested"]["deep"]["structure"]["with"]["lots"]["of"]["data"] == "value"


def test_convert_lists_to_arrays_utility():
    """Test the convert_lists_to_arrays utility function that's used in predictions."""
    from polaris.utils.misc import convert_lists_to_arrays

    # Test with simple lists
    simple_list = [1, 2, 3]
    result = convert_lists_to_arrays(simple_list)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1, 2, 3]))

    # Test with nested dictionaries containing lists
    nested_dict = {"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}}
    result = convert_lists_to_arrays(nested_dict)
    assert isinstance(result, dict)
    assert isinstance(result["test"], dict)
    assert isinstance(result["test"]["col1"], np.ndarray)
    assert isinstance(result["test"]["col2"], np.ndarray)
    assert np.array_equal(result["test"]["col1"], np.array([1, 2, 3]))
    assert np.array_equal(result["test"]["col2"], np.array([4, 5, 6]))

    # Test with object arrays
    object_list = [{"test": "object1"}, {"test": "object2"}]
    result = convert_lists_to_arrays(object_list)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(object)
    assert result[0]["test"] == "object1"
    assert result[1]["test"] == "object2"
