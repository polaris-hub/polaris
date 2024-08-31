from polaris.benchmark.predictions import BenchmarkPredictions
import pytest

def test_benchmark_predictions_initialization():
    "Single task, single test set"
    assert {"test": {"col1": [1, 2, 3]}} == BenchmarkPredictions(
        predictions=[1, 2, 3], target_cols=["col1"]).predictions
    assert {"test": {"col1": [1, 2, 3]}} == BenchmarkPredictions(
        predictions={"col1": [1, 2, 3]}, target_cols=["col1"]
    ).predictions
    assert {"test": {"col1": [1, 2, 3]}} == BenchmarkPredictions(
        predictions={"test": {"col1": [1, 2, 3]}}, target_cols=["col1"]
    ).predictions

    "Single task, multiple test sets"
    assert {"test": {"col1": [1, 2, 3]},
            "test2": {"col1": [4, 5, 6]}} == BenchmarkPredictions(
        predictions={"test": [1, 2, 3], "test2": [4, 5, 6]}, target_cols=["col1"]
    ).predictions
    assert {"test1": {"col1": [1, 2, 3]},
            "test2": {"col1": [4, 5, 6]}} == BenchmarkPredictions(
        predictions={"test1": {"col1": [1, 2, 3]}, "test2": {"col1": [4, 5, 6]}},
        target_cols=["col1"]
    ).predictions

    "Multi-task, single test set"
    assert {"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}} == BenchmarkPredictions(
        predictions={"col1": [1, 2, 3], "col2": [4, 5, 6]},
        target_cols=["col1", "col2"]
    ).predictions
    assert {"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}} == BenchmarkPredictions(
        predictions={"test": {"col1": [1, 2, 3], "col2": [4, 5, 6]}},
        target_cols=["col1", "col2"]
    ).predictions

    "Multi-task, multiple test sets"
    assert {
        "test1": {"col1": [1, 2, 3], "col2": [4, 5, 6]},
        "test2": {"col1": [7, 8, 9], "col2": [10, 11, 12]}
    } == BenchmarkPredictions(
        predictions={"test1": {"col1": [1, 2, 3], "col2": [4, 5, 6]},
                     "test2": {"col1": [7, 8, 9], "col2": [10, 11, 12]}},
        target_cols=["col1", "col2"]
    ).predictions

    # "Invalid predictions"
    # with pytest.raises(ValueError,
    #                    match="Invalid structure for test set 'test'."
    #                    "Expected a dictionary of \\{col_name: predictions\\}"):
    #     BenchmarkPredictions(predictions={"test": [1, 2, 3]}, target_cols=["col1"])

    # with pytest.raises(ValueError,
    #                    match="Invalid predictions for test set 'test', target 'col1'."
    #                    "Expected a numpy array or list."):
    #     BenchmarkPredictions(predictions={"test": {"col1": "not an array"}},
    #                          target_cols=["col1"])


# def test_benchmark_predictions_to_dataframe():
#     y_pred = {"target1": np.array([0, 1, 0]), "target2": np.array([1, 1, 0])}
#     y_prob = {"target1": np.array([0.2, 0.8, 0.3]), "target2": np.array([0.9, 0.7, 0.4])}

#     predictions = BenchmarkPredictions(y_pred=y_pred, y_prob=y_prob)

#     df = predictions.to_dataframe()

#     assert isinstance(df, pd.DataFrame)
#     assert set(df.columns) == {"target1_pred", "target1_prob", "target2_pred", "target2_prob"}
#     assert df.shape == (3, 4)

#     # Test with only y_pred
#     predictions_only_pred = BenchmarkPredictions(y_pred=y_pred)
#     df_only_pred = predictions_only_pred.to_dataframe()

#     assert isinstance(df_only_pred, pd.DataFrame)
#     assert set(df_only_pred.columns) == {"target1_pred", "target2_pred"}
#     assert df_only_pred.shape == (3, 2)

#     # Test with only y_prob
#     predictions_only_prob = BenchmarkPredictions(y_prob=y_prob)
#     df_only_prob = predictions_only_prob.to_dataframe()

#     assert isinstance(df_only_prob, pd.DataFrame)
#     assert set(df_only_prob.columns) == {"target1_prob", "target2_prob"}
#     assert df_only_prob.shape == (3, 2)


# def test_benchmark_predictions_from_dataframe():
#     df = pd.DataFrame({
#         "target1_pred": [0, 1, 0],
#         "target1_prob": [0.2, 0.8, 0.3],
#         "target2_pred": [1, 1, 0],
#         "target2_prob": [0.9, 0.7, 0.4]
#     })

#     predictions = BenchmarkPredictions.from_dataframe(df)

#     assert set(predictions.y_pred.keys()) == {"target1", "target2"}
#     assert set(predictions.y_prob.keys()) == {"target1", "target2"}
#     assert np.array_equal(predictions.y_pred["target1"], np.array([0, 1, 0]))
#     assert np.array_equal(predictions.y_prob["target2"], np.array([0.9, 0.7, 0.4]))

#     # Test with only pred columns
#     df_only_pred = df.drop(["target1_prob", "target2_prob"], axis=1)
#     predictions_only_pred = BenchmarkPredictions.from_dataframe(df_only_pred)

#     assert set(predictions_only_pred.y_pred.keys()) == {"target1", "target2"}
#     assert predictions_only_pred.y_prob is None

#     # Test with only prob columns
#     df_only_prob = df.drop(["target1_pred", "target2_pred"], axis=1)
#     predictions_only_prob = BenchmarkPredictions.from_dataframe(df_only_prob)

#     assert predictions_only_prob.y_pred is None
#     assert set(predictions_only_prob.y_prob.keys()) == {"target1", "target2"}

#     # Test with invalid column names
#     df_invalid = pd.DataFrame({"invalid_col": [1, 2, 3]})
#     with pytest.raises(ValueError, match="No valid prediction columns found"):
#         BenchmarkPredictions.from_dataframe(df_invalid)
