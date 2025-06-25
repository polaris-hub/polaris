from polaris.prediction._predictions_v2 import BenchmarkPredictionsV2
from polaris.utils.zarr.codecs import RDKitMolCodec, AtomArrayCodec
from rdkit import Chem
import numpy as np
import pytest
import datamol as dm
import zarr
from fastpdb import struc


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


def test_v2_rdkit_object_codec(v2_benchmark_with_rdkit_object_dtype):
    mols = [dm.to_mol("CCO"), dm.to_mol("CCN")]
    preds = {"test": {"expt": mols}}
    bp = BenchmarkPredictionsV2(
        predictions=preds,
        dataset_zarr_root=v2_benchmark_with_rdkit_object_dtype.dataset.zarr_root,
        benchmark_artifact_id=v2_benchmark_with_rdkit_object_dtype.artifact_id,
        target_labels=["expt"],
        test_set_labels=["test"],
        test_set_sizes={"test": 2},
    )
    assert isinstance(bp.predictions["test"]["expt"], np.ndarray)
    assert bp.predictions["test"]["expt"].dtype == object
    assert_deep_equal(bp.predictions, {"test": {"expt": np.array(mols, dtype=object)}})

    # Check Zarr archive
    zarr_path = bp.to_zarr()
    assert zarr_path.exists()
    root = zarr.open(str(zarr_path), mode="r")
    arr = root["test"]["expt"][:]
    arr_smiles = [Chem.MolToSmiles(m) for m in arr]
    mols_smiles = [Chem.MolToSmiles(m) for m in mols]
    assert arr_smiles == mols_smiles

    # Check that object_codec is correctly set as a filter (Zarr stores object_codec as filters)
    zarr_array = root["test"]["expt"]
    assert zarr_array.filters is not None
    assert len(zarr_array.filters) > 0
    assert any(isinstance(f, RDKitMolCodec) for f in zarr_array.filters)


def test_v2_atomarray_object_codec(v2_benchmark_with_atomarray_object_dtype, pdbs_structs):
    # Use fastpdb.AtomArray objects
    preds = {"test": {"expt": np.array(pdbs_structs[:2], dtype=object)}}
    bp = BenchmarkPredictionsV2(
        predictions=preds,
        dataset_zarr_root=v2_benchmark_with_atomarray_object_dtype.dataset.zarr_root,
        benchmark_artifact_id=v2_benchmark_with_atomarray_object_dtype.artifact_id,
        target_labels=["expt"],
        test_set_labels=["test"],
        test_set_sizes={"test": 2},
    )
    assert isinstance(bp.predictions["test"]["expt"], np.ndarray)
    assert bp.predictions["test"]["expt"].dtype == object
    assert_deep_equal(bp.predictions, {"test": {"expt": np.array(pdbs_structs[:2], dtype=object)}})

    # Check Zarr archive (dtype and shape only)
    zarr_path = bp.to_zarr()
    assert zarr_path.exists()
    root = zarr.open(str(zarr_path), mode="r")
    arr = root["test"]["expt"][:]
    assert arr.dtype == object
    assert arr.shape == (2,)
    assert all(isinstance(x, struc.AtomArray) for x in arr)

    # Check that object_codec is correctly set as a filter (Zarr stores object_codec as filters)
    zarr_array = root["test"]["expt"]
    assert zarr_array.filters is not None
    assert len(zarr_array.filters) > 0
    assert any(isinstance(f, AtomArrayCodec) for f in zarr_array.filters)


def test_v2_dtype_mismatch_raises(test_benchmark_v2):
    # Create a list of rdkit.Chem.Mol objects (object dtype) to test against float dtype dataset
    mols = [dm.to_mol("CCO"), dm.to_mol("CCN")]
    preds = {"test": {"A": mols}}  # Using column "A" which has float dtype in test_dataset_v2
    with pytest.raises(ValueError, match="Dtype mismatch"):
        BenchmarkPredictionsV2(
            predictions=preds,
            dataset_zarr_root=test_benchmark_v2.dataset.zarr_root,
            benchmark_artifact_id=test_benchmark_v2.artifact_id,
            target_labels=["A"],
            test_set_labels=["test"],
            test_set_sizes={"test": 2},
        )
