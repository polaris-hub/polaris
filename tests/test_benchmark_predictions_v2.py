from polaris.prediction._predictions_v2 import BenchmarkPredictionsV2
from rdkit import Chem

BenchmarkPredictionsV2.model_rebuild()

import numpy as np
import pytest
import datamol as dm
import numcodecs
import zarr
from polaris.dataset import DatasetV2, ColumnAnnotation
from polaris.benchmark._split_v2 import SplitV2, IndexSet
from polaris.benchmark._benchmark_v2 import BenchmarkV2Specification

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

@pytest.fixture(scope="function")
def v2_benchmark_with_object_dtype(tmp_path, test_org_owner):
    zarr_path = tmp_path / "test_object_dtype.zarr"
    root = zarr.open(str(zarr_path), mode="w")
    root.array(
        "expt",
        data=np.empty(2, dtype=object),
        dtype=object,
        object_codec=numcodecs.Pickle(),
    )
    zarr.consolidate_metadata(root.store)
    dataset = DatasetV2(
        name="test-dataset-v2-object",
        source="https://www.example.com",
        annotations={"expt": ColumnAnnotation(user_attributes={"unit": "kcal/mol"})},
        tags=["tagA", "tagB"],
        user_attributes={"attributeA": "valueA", "attributeB": "valueB"},
        owner=test_org_owner,
        license="CC-BY-4.0",
        curation_reference="https://www.example.com",
        zarr_root_path=str(zarr_path),
    )
    train_indices = [0]
    test_indices = [1]
    split = SplitV2(training=IndexSet(indices=train_indices), test=IndexSet(indices=test_indices))
    benchmark = BenchmarkV2Specification(
        name="v2-benchmark-object-dtype",
        dataset=dataset,
        metrics=["mean_absolute_error"],
        main_metric="mean_absolute_error",
        split=split,
        target_cols=["expt"],
        input_cols=["smiles"],
    )
    return benchmark


@pytest.fixture(scope="function")
def v2_benchmark_with_float_dtype(tmp_path, test_org_owner):
    zarr_path = tmp_path / "test_float_dtype.zarr"
    root = zarr.open(str(zarr_path), mode="w")
    root.array(
        "expt",
        data=np.zeros(2, dtype=float),
        dtype=float,
        object_codec=None,
    )
    zarr.consolidate_metadata(root.store)
    dataset = DatasetV2(
        name="test-dataset-v2-float",
        source="https://www.example.com",
        annotations={"expt": ColumnAnnotation(user_attributes={"unit": "kcal/mol"})},
        tags=["tagA", "tagB"],
        user_attributes={"attributeA": "valueA", "attributeB": "valueB"},
        owner=test_org_owner,
        license="CC-BY-4.0",
        curation_reference="https://www.example.com",
        zarr_root_path=str(zarr_path),
    )
    train_indices = [0]
    test_indices = [1]
    split = SplitV2(training=IndexSet(indices=train_indices), test=IndexSet(indices=test_indices))
    benchmark = BenchmarkV2Specification(
        name="v2-benchmark-float-dtype",
        dataset=dataset,
        metrics=["mean_absolute_error"],
        main_metric="mean_absolute_error",
        split=split,
        target_cols=["expt"],
        input_cols=["smiles"],
    )
    return benchmark


def test_v2_normalization_and_object_dtype(v2_benchmark_with_object_dtype):
    # Create a list of rdkit.Chem.Mol objects
    mols = [dm.to_mol("CCO"), dm.to_mol("CCN")]
    preds = {"test": {"expt": mols}}
    bp = BenchmarkPredictionsV2(
        predictions=preds,
        benchmark=v2_benchmark_with_object_dtype,
        target_labels=["expt"],
        test_set_labels=["test"],
        test_set_sizes={"test": 2},
    )
    assert isinstance(bp.predictions["test"]["expt"], np.ndarray)
    assert bp.predictions["test"]["expt"].dtype == object
    assert_deep_equal(bp.predictions, {"test": {"expt": np.array(mols, dtype=object)}})


def test_v2_fastpdb_object_dtype(v2_benchmark_with_object_dtype, pdbs_structs):
    # Use fastpdb.AtomArray objects
    preds = {"test": {"expt": np.array(pdbs_structs[:2], dtype=object)}}    
    bp = BenchmarkPredictionsV2(
        predictions=preds,
        benchmark=v2_benchmark_with_object_dtype,
        target_labels=["expt"],
        test_set_labels=["test"],
        test_set_sizes={"test": 2},
    )
    assert isinstance(bp.predictions["test"]["expt"], np.ndarray)
    assert bp.predictions["test"]["expt"].dtype == object
    assert_deep_equal(bp.predictions, {"test": {"expt": np.array(pdbs_structs[:2], dtype=object)}})


def test_v2_dtype_mismatch_raises(v2_benchmark_with_float_dtype):
    # Create a list of rdkit.Chem.Mol objects
    mols = [dm.to_mol("CCO"), dm.to_mol("CCN")]
    preds = {"test": {"expt": mols}}
    with pytest.raises(ValueError, match="Dtype mismatch"):
        BenchmarkPredictionsV2(
            predictions=preds,
            benchmark=v2_benchmark_with_float_dtype,
            target_labels=["expt"],
            test_set_labels=["test"],
            test_set_sizes={"test": 2},
        )


def test_to_zarr_creates_archive(v2_benchmark_with_object_dtype):
    mols = [dm.to_mol("CCO"), dm.to_mol("CCN")]
    preds = {"test": {"expt": mols}}
    bp = BenchmarkPredictionsV2(
        predictions=preds,
        benchmark=v2_benchmark_with_object_dtype,
        target_labels=["expt"],
        test_set_labels=["test"],
        test_set_sizes={"test": 2},
    )
    zarr_path = bp.to_zarr()
    assert zarr_path.exists()
    root = zarr.open(str(zarr_path), mode="r")
    assert "test" in root
    assert "expt" in root["test"]
    arr = root["test"]["expt"][:]
    assert arr.dtype == object
    arr_smiles = [Chem.MolToSmiles(m) for m in arr]
    mols_smiles = [Chem.MolToSmiles(m) for m in mols]
    assert arr_smiles == mols_smiles
