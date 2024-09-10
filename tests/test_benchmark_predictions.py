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


def test_benchmark_predictions_normalization():
    # Single task, single test set
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}},
        BenchmarkPredictions(
            predictions=[1, 2, 3], target_cols=["col1"], test_set_names=["test"]
        ).predictions,
    )
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}},
        BenchmarkPredictions(
            predictions={"col1": [1, 2, 3]}, target_cols=["col1"], test_set_names=["test"]
        ).predictions,
    )
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}},
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3]}}, target_cols=["col1"], test_set_names=["test"]
        ).predictions,
    )

    # Single task, multiple test sets
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3])}, "test2": {"col1": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test": [1, 2, 3], "test2": [4, 5, 6]},
            target_cols=["col1"],
            test_set_names=["test", "test2"],
        ).predictions,
    )
    assert_deep_equal(
        {"test1": {"col1": np.array([1, 2, 3])}, "test2": {"col1": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test1": {"col1": [1, 2, 3]}, "test2": {"col1": [4, 5, 6]}},
            target_cols=["col1"],
            test_set_names=["test1", "test2"],
        ).predictions,
    )

    # Multi-task, single test set
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3]), "col2": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"col1": [1, 2, 3], "col2": [4, 5, 6]},
            target_cols=["col1", "col2"],
            test_set_names=["test"],
        ).predictions,
    )
    assert_deep_equal(
        {"test": {"col1": np.array([1, 2, 3]), "col2": np.array([4, 5, 6])}},
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}},
            target_cols=["col1", "col2"],
            test_set_names=["test"],
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
            test_set_names=["test1", "test2"],
        ).predictions,
    )


def test_benchmark_predictions_correct_keys():
    with pytest.raises(
        ValueError,
        match="Missing predictions. Single array/list of predictions provided "
        "but multiple test sets expected with names \['test1', 'test2'\].",
    ):
        BenchmarkPredictions(
            predictions=[1, 2, 3],
            target_cols=["col1"],
            test_set_names=["test1", "test2"],
        )

    with pytest.raises(
        ValueError,
        match="Missing targets. Single array/list of predictions provided "
        "but multiple targets expected with names \['col1', 'col2'\].",
    ):
        BenchmarkPredictions(
            predictions=[1, 2, 3],
            target_cols=["col1", "col2"],
            test_set_names=["test1"],
        )

    with pytest.raises(
        ValueError,
        match="Missing test sets. Single dictionary of predictions provided "
        "but multiple test sets expected with names \['test1', 'test2'\].",
    ):
        BenchmarkPredictions(
            predictions={"col1": [1, 2, 3]},
            target_cols=["col1"],
            test_set_names=["test1", "test2"],
        )

    with pytest.raises(
        ValueError,
        match="Predictions must be provided for all test sets: \['test1', 'test2', 'test3'\].",
    ):
        BenchmarkPredictions(
            predictions={"test1": {"col1": [1, 2, 3]}, "test2": {"col1": [4, 5, 6]}},
            target_cols=["col1"],
            test_set_names=["test1", "test2", "test3"],
        )


def test_benchmark_predictions_type_checking():
    assert {"test": {"col1": ["strings", "also", "valid"]}} == BenchmarkPredictions(
        predictions=["strings", "also", "valid"], target_cols=["col1"], test_set_names=["test"]
    ).predictions


def test_invalid_benchmark_predictions_errors():
    with pytest.raises(
        ValueError,
        match="Invalid structure for test set 'test2'. " "Expected a dictionary of {col_name: predictions}",
    ):
        BenchmarkPredictions(
            predictions={"test": {"col1": [1, 2, 3]}, "test2": [4, 5, 6]},
            target_cols=["col1", "col2"],
            test_set_names=["test", "test2"],
        )

    with pytest.raises(
        ValueError,
        match="Invalid predictions for test set 'test', target 'col1'. " "Expected a numpy array or list.",
    ):
        BenchmarkPredictions(
            predictions={"test": {"col1": "not an array or list"}},
            target_cols=["col1"],
            test_set_names=["test"],
        )

    with pytest.raises(
        ValueError,
        match="Invalid predictions for test set 'test'. Target 'wrong column name' is not in target_cols: \['col1'\].",
    ):
        BenchmarkPredictions(
            predictions={"test": {"wrong column name": [1, 2, 3]}},
            target_cols=["col1"],
            test_set_names=["test"],
        )


def test_benchmark_predictions_serialization():
    predictions = BenchmarkPredictions(predictions=[1, 2, 3], target_cols=["col1"], test_set_names=["test"])
    serialized = predictions.model_dump()
    assert serialized["predictions"] == {"test": {"col1": [1, 2, 3]}}
    assert serialized["target_cols"] == ["col1"]
