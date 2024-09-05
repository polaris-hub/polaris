from polaris.benchmark.predictions import BenchmarkPredictions
import numpy as np
import pytest


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


def test_benchmark_predictions_initialization():
    # Single task, single test set
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}},
        BenchmarkPredictions(predictions=[1, 2, 3], target_cols=["col1"]).predictions,
    )
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}},
        BenchmarkPredictions(predictions={"col1": [1, 2, 3]}, target_cols=["col1"]).predictions,
    )
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}},
        BenchmarkPredictions(predictions={"test": {"col1": [1, 2, 3]}}, target_cols=["col1"]).predictions,
    )

    # Single task, multiple test sets
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}, "test2": {"col1": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test": [1, 2, 3], "test2": [4, 5, 6]}, target_cols=["col1"]
        ).predictions,
    )
    assert_deep_equal(
        {"test1": {"col1": np.array([1, 2, 3])}, "test2": {"col1": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test1": {"col1": [1, 2, 3]}, "test2": {"col1": [4, 5, 6]}}, target_cols=["col1"]
        ).predictions,
    )

    # Multi-task, single test set
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3]), "col2": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"col1": [1, 2, 3], "col2": [4, 5, 6]}, target_cols=["col1", "col2"]
        ).predictions,
    )
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3]), "col2": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}}, target_cols=["col1", "col2"]
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
            target_cols=["col1", "col2"],
        ).predictions,
    )

    "Invalid predictions"
    with pytest.raises(
        ValueError,
        match="Invalid structure for test set 'test2'. " "Expected a dictionary of {col_name: predictions}",
    ):
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3]}, "test2": [4, 5, 6]}, target_cols=["col1", "col2"]
        )

    with pytest.raises(
        ValueError,
        match="Invalid predictions for test set 'test', target 'col1'. "
        "Expected a numpy array or list of numbers.",
    ):
        BenchmarkPredictions(predictions={"test": {"col1": "not an array"}}, target_cols=["col1"])

    with pytest.raises(
        ValueError,
        match="Invalid predictions for test set 'test'. Target 'wrong column name' "
        "is not in target_cols: ['col1'].",
    ):
        BenchmarkPredictions(predictions={"test": {"wrong column name": [1, 2, 3]}}, target_cols=["col1"])


def test_benchmark_predictions_serialization():
    predictions = BenchmarkPredictions(predictions=[1, 2, 3], target_cols=["col1"])
    serialized = predictions.model_dump()
    assert serialized["predictions"] == {"test": {"col1": [1, 2, 3]}}
    assert serialized["target_cols"] == ["col1"]
