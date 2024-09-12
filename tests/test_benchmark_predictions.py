import numpy as np
import pytest

from polaris.evaluate import BenchmarkPredictions


def assert_deep_equal(result, expected):
    assert isinstance(result, type(expected)), f"Types differ: {type(result)} != {type(expected)}"

    if isinstance(expected, dict):
        assert result.keys() == expected.keys()
        for key in expected:
            assert_deep_equal(result[key], expected[key])
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


def test_benchmark_predictions_normalization():
    # Single task, single test set
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}},
        BenchmarkPredictions(
            predictions=[1, 2, 3], target_labels=["col1"], test_set_labels=["test"]
        ).predictions,
    )
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}},
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3]}}, target_labels=["col1"], test_set_labels=["test"]
        ).predictions,
    )

    # Single task, multiple test sets
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}, "test2": {"col1": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test": [1, 2, 3], "test2": [4, 5, 6]},
            target_labels=["col1"],
            test_set_labels=["test", "test2"],
        ).predictions,
    )
    assert_deep_equal(
        {"test1": {"col1": np.array([1, 2, 3])}, "test2": {"col1": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test1": {"col1": [1, 2, 3]}, "test2": {"col1": [4, 5, 6]}},
            target_labels=["col1"],
            test_set_labels=["test1", "test2"],
        ).predictions,
    )

    # Multi-task, single test set
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3]), "col2": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"col1": [1, 2, 3], "col2": [4, 5, 6]},
            target_labels=["col1", "col2"],
            test_set_labels=["test"],
        ).predictions,
    )
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3]), "col2": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}},
            target_labels=["col1", "col2"],
            test_set_labels=["test"],
        ).predictions,
    )

    # Multi-task, multiple test sets
    assert_deep_equal(
        {
            "test1": {"col1": np.array([1, 2, 3]), "col2": np.array([4, 5, 6])},
            "test2": {"col1": np.array([7, 8, 9]), "col2": np.array([10, 11, 12])},
        },
        BenchmarkPredictions(
            predictions={
                "test1": {"col1": [1, 2, 3], "col2": [4, 5, 6]},
                "test2": {"col1": [7, 8, 9], "col2": [10, 11, 12]},
            },
            target_labels=["col1", "col2"],
            test_set_labels=["test1", "test2"],
        ).predictions,
    )


def test_benchmark_predictions_incorrect_keys():
    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions=[1, 2, 3],
            target_labels=["col1"],
            test_set_labels=["test1", "test2"],
        )

    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions=[1, 2, 3],
            target_labels=["col1", "col2"],
            test_set_labels=["test1"],
        )

    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"col1": [1, 2, 3]},
            target_labels=["col1"],
            test_set_labels=["test1", "test2"],
        )

    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"test1": {"col1": [1, 2, 3]}, "test2": {"col1": [4, 5, 6]}},
            target_labels=["col1"],
            test_set_labels=["test1", "test2", "test3"],
        )


def test_benchmark_predictions_type_checking():
    v1 = {"test": {"col1": ["strings", "also", "valid"]}}
    v2 = BenchmarkPredictions(
        predictions=["strings", "also", "valid"], target_labels=["col1"], test_set_labels=["test"]
    ).predictions

    assert list(v1.keys()) == ["test"]
    assert list(v2.keys()) == ["test"]
    assert list(v1["test"].keys()) == ["col1"]
    assert list(v2["test"].keys()) == ["col1"]
    assert isinstance(v2["test"]["col1"], np.ndarray)
    assert np.array_equal(v2["test"]["col1"], np.array(["strings", "also", "valid"]))


def test_invalid_benchmark_predictions_errors():
    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3]}, "test2": [4, 5, 6]},
            target_cols=["col1", "col2"],
            test_set_labels=["test", "test2"],
        )

    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"test": {"col1": "not an array or list"}},
            target_cols=["col1"],
            test_set_labels=["test"],
        )

    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"test": {"wrong column name": [1, 2, 3]}},
            target_cols=["col1"],
            test_set_labels=["test"],
        )

    # You should either fully or minimally specify the predictions.
    # We don't allow in-between results.
    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"col1": [1, 2, 3]},
            target_cols=["col1"],
            test_set_labels=["test"],
        )

    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"test": [1, 2, 3]},
            target_cols=["col1"],
            test_set_labels=["test"],
        )

    # You shouldn't specify more keys than expected.
    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}},
            target_cols=["col1"],
            test_set_labels=["test"],
        )
    with pytest.raises(ValueError):
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3]}, "test2": {"col1": [1, 2, 3]}},
            target_cols=["col1"],
            test_set_labels=["test"],
        )


def test_benchmark_predictions_serialization():
    predictions = BenchmarkPredictions(
        predictions=[1, 2, 3], target_labels=["col1"], test_set_labels=["test"]
    )
    serialized = predictions.model_dump()
    assert serialized["predictions"] == {"test": {"col1": [1, 2, 3]}}
    assert serialized["target_labels"] == ["col1"]
    assert serialized["test_set_labels"] == ["test"]

    deserialized = BenchmarkPredictions(**serialized)
    assert set(deserialized.predictions.keys()) == {"test"}
    assert set(deserialized.predictions["test"].keys()) == {"col1"}
    assert np.array_equal(deserialized.predictions["test"]["col1"], np.array([1, 2, 3]))
    assert deserialized.target_labels == ["col1"]
    assert deserialized.test_set_labels == ["test"]
