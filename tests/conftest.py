import biotite.database.rcsb as rcsb
import datamol as dm
import fastpdb
import numpy as np
import pytest
import zarr
from datamol.utils import fs

import polaris as po
from polaris.benchmark import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.competition import CompetitionSpecification
from polaris.dataset import ColumnAnnotation, DatasetFactory, DatasetV1, DatasetV2
from polaris.dataset.converters import SDFConverter
from polaris.utils.types import HubOwner
from polaris.benchmark._benchmark_v2 import BenchmarkV2Specification
from polaris.benchmark._split_v2 import SplitV2, IndexSet


def check_version(artifact):
    assert po.__version__ == artifact.polaris_version


@pytest.fixture(scope="module")
def test_data():
    data = dm.data.freesolv()[:100]
    # set an abitrary threshold for testing purpose.
    data["CLASS_expt"] = data["expt"].gt(0).astype(int).values
    data["CLASS_calc"] = data["calc"].gt(0).astype(int).values
    data["MULTICLASS_expt"] = np.random.randint(low=0, high=3, size=data.shape[0])
    data["MULTICLASS_calc"] = np.random.randint(low=0, high=3, size=data.shape[0])
    return data


@pytest.fixture(scope="module")
def caffeine():
    # Let's generate a toy dataset with a single molecule
    smiles = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    mol = dm.to_mol(smiles)

    # We will generate 3D conformers for this molecule with some conformers
    # NOTE (cwognum): We only generate a single conformer, because dm.to_sdf() only saves one.
    mol = dm.conformers.generate(mol, align_conformers=True, n_confs=1)

    # Let's also set a molecular property
    mol.SetProp("my_property", "my_value_1")
    return mol


@pytest.fixture(scope="module")
def ibuprofen():
    # Let's generate a toy dataset with a single molecule
    smiles = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"
    mol = dm.to_mol(smiles)

    # We will generate 3D conformers for this molecule with some conformers
    # NOTE (cwognum): We only generate a single conformer, because dm.to_sdf() only saves one.
    mol = dm.conformers.generate(mol, align_conformers=True, n_confs=1)

    # Let's also set a molecular property
    mol.SetProp("my_property", "my_value_2")
    return mol


@pytest.fixture(scope="module")
def pdb_paths(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    pdb_paths = rcsb.fetch(["1l2y", "4i23"], "pdb", tmp_dir)
    return pdb_paths


@pytest.fixture(scope="module")
def pdbs_structs(pdb_paths):
    pdb_arrays = []
    for pdb_path in pdb_paths:
        in_file = fastpdb.PDBFile.read(pdb_path)
        atom_array = in_file.get_structure(
            model=1, include_bonds=True, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
        )
        pdb_arrays.append(atom_array)

    return pdb_arrays


@pytest.fixture(scope="module")
def sdf_files(tmp_path_factory, caffeine, ibuprofen):
    path_1 = tmp_path_factory.mktemp("data") / "caffeine.sdf"
    path_2 = tmp_path_factory.mktemp("data") / "ibuprofen.sdf"
    dm.to_sdf(caffeine, path_1)
    dm.to_sdf(ibuprofen, path_2)
    return [path_1, path_2]


@pytest.fixture(scope="module")
def sdf_file(tmp_path_factory, caffeine):
    path = tmp_path_factory.mktemp("data") / "caffeine.sdf"
    dm.to_sdf(caffeine, path)
    return path


@pytest.fixture(scope="module")
def test_org_owner():
    return HubOwner(organizationId="test-organization", slug="test-organization")


@pytest.fixture(scope="module")
def test_user_owner():
    return HubOwner(userId="test-user", slug="test-user")


@pytest.fixture(scope="function")
def test_dataset(test_data, test_org_owner) -> DatasetV1:
    dataset = DatasetV1(
        table=test_data,
        name="test-dataset",
        source="https://www.example.com",
        annotations={"expt": ColumnAnnotation(user_attributes={"unit": "kcal/mol"})},
        tags=["tagA", "tagB"],
        user_attributes={"attributeA": "valueA", "attributeB": "valueB"},
        owner=test_org_owner,
        license="CC-BY-4.0",
        curation_reference="https://www.example.com",
    )
    check_version(dataset)
    return dataset


@pytest.fixture(scope="function")
def test_dataset_v2(zarr_archive, test_org_owner) -> DatasetV2:
    dataset = DatasetV2(
        name="test-dataset-v2",
        source="https://www.example.com",
        annotations={"A": ColumnAnnotation(user_attributes={"unit": "kcal/mol"})},
        tags=["tagA", "tagB"],
        user_attributes={"attributeA": "valueA", "attributeB": "valueB"},
        owner=test_org_owner,
        license="CC-BY-4.0",
        curation_reference="https://www.example.com",
        zarr_root_path=zarr_archive,
    )
    check_version(dataset)
    return dataset


@pytest.fixture(scope="function")
def zarr_archive(tmp_path):
    tmp_path = fs.join(tmp_path, "data.zarr")
    root = zarr.open(tmp_path, mode="w")
    root.array("A", data=np.random.random((100, 2048)), chunks=(1, None))
    root.array("B", data=np.random.random((100, 2048)), chunks=(1, None))
    zarr.consolidate_metadata(root.store)
    return tmp_path


@pytest.fixture()
def regression_metrics():
    return [
        "mean_absolute_error",
        "mean_squared_error",
        "r2",
        "spearmanr",
        "pearsonr",
        "explained_var",
        "absolute_average_fold_error",
    ]


@pytest.fixture()
def classification_metrics():
    return [
        "accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "cohen_kappa",
        "balanced_accuracy",
    ]


@pytest.fixture(scope="function")
def test_single_task_benchmark(test_dataset, regression_metrics):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-single-set-benchmark",
        dataset=test_dataset,
        metrics=regression_metrics,
        main_metric="mean_absolute_error",
        split=(train_indices, test_indices),
        target_cols="expt",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_single_task_benchmark_clf(test_dataset, classification_metrics):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-single-set-benchmark",
        dataset=test_dataset,
        main_metric="accuracy",
        metrics=classification_metrics,
        split=(train_indices, test_indices),
        target_cols="CLASS_expt",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_single_task_benchmark_multi_clf(test_dataset, classification_metrics):
    np.random.seed(111)
    indices = np.arange(100)
    np.random.shuffle(indices)
    train_indices = indices[:80]
    test_indices = indices[80:]

    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-single-set-benchmark",
        dataset=test_dataset,
        main_metric="accuracy",
        metrics=[
            "accuracy",
            "balanced_accuracy",
            "mcc",
            "cohen_kappa",
            "f1_macro",
            "f1_micro",
            "roc_auc_ovr",
            "roc_auc_ovo",
            "pr_auc",
        ],
        split=(train_indices, test_indices),
        target_cols="MULTICLASS_expt",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_single_task_benchmark_multiple_test_sets(test_dataset, regression_metrics):
    train_indices = list(range(90))
    test_indices = {"test_1": list(range(90, 95)), "test_2": list(range(95, 100))}
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-multi-set-benchmark",
        dataset=test_dataset,
        metrics=regression_metrics,
        main_metric="r2",
        split=(train_indices, test_indices),
        target_cols="expt",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_single_task_benchmark_clf_multiple_test_sets(test_dataset, classification_metrics):
    np.random.seed(111)  # make sure two classes in `y_true`
    indices = np.arange(100)
    np.random.shuffle(indices)
    train_indices = indices[:80]
    test_indices = {"test_1": indices[80:90], "test_2": indices[90:]}
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-multi-set-benchmark-clf",
        dataset=test_dataset,
        metrics=classification_metrics,
        main_metric="pr_auc",
        split=(train_indices, test_indices),
        target_cols="CLASS_calc",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_multi_task_benchmark(test_dataset, regression_metrics):
    # For the sake of simplicity, just use a small set of indices
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    benchmark = MultiTaskBenchmarkSpecification(
        name="multi-task-benchmark",
        dataset=test_dataset,
        main_metric="mean_absolute_error",
        metrics=regression_metrics,
        split=(train_indices, test_indices),
        target_cols=["expt", "calc"],
        input_cols="smiles",
        target_types={"expt": "regression"},
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_multi_task_benchmark_clf(test_dataset, classification_metrics):
    # For the sake of simplicity, just use a small set of indices
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    benchmark = MultiTaskBenchmarkSpecification(
        name="multi-task-benchmark",
        dataset=test_dataset,
        main_metric="accuracy",
        metrics=classification_metrics,
        split=(train_indices, test_indices),
        target_cols=["CLASS_expt", "CLASS_calc"],
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_competition(zarr_archive, test_org_owner):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    competition = CompetitionSpecification(
        # Base attributes
        name="test-competition",
        owner=test_org_owner,
        tags=["tagA", "tagB"],
        user_attributes={"attributeA": "valueA", "attributeB": "valueB"},
        # Benchmark attributes
        split=(train_indices, test_indices),
        input_cols=["A"],
        target_cols=["B"],
        readme="Testing specification",
        # Dataset attributes
        source="https://www.example.com",
        annotations={"A": ColumnAnnotation(user_attributes={"unit": "kcal/mol"})},
        license="CC-BY-4.0",
        curation_reference="https://www.example.com",
        zarr_root_path=zarr_archive,
        # Competition attributes
        start_time="2025-01-13T21:59:38Z",
        end_time="2025-01-20T21:59:38Z",
        n_classes={"B": 0},
    )
    check_version(competition)
    return competition


@pytest.fixture(scope="function")
def test_multi_task_benchmark_multiple_test_sets(test_dataset, regression_metrics):
    train_indices = list(range(90))
    test_indices = {"test_1": list(range(90, 95)), "test_2": list(range(95, 100))}
    benchmark = MultiTaskBenchmarkSpecification(
        name="multi-task-multi-set-benchmark",
        dataset=test_dataset,
        metrics=regression_metrics,
        main_metric="r2",
        split=(train_indices, test_indices),
        target_cols=["expt", "calc"],
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_docking_dataset(tmp_path, sdf_files, test_org_owner):
    # toy docking dataset
    factory = DatasetFactory(str(tmp_path / "ligands.zarr"))

    converter = SDFConverter(mol_prop_as_cols=True)
    factory.register_converter("sdf", converter)
    factory.add_from_files(sdf_files, axis=0)
    dataset = factory.build()
    check_version(dataset)
    return dataset


@pytest.fixture(scope="function")
def test_docking_benchmark(test_docking_dataset):
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-single-set-benchmark",
        dataset=test_docking_dataset,
        metrics=["rmsd_coverage"],
        main_metric="rmsd_coverage",
        split=([], [0, 1]),
        target_cols=["molecule"],
        input_cols=["smiles"],
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_benchmark_v2(test_dataset_v2, test_org_owner):
    from polaris.benchmark._split_v2 import TrainTestIndices

    train_indices = [0]
    test_indices = [1]
    split = SplitV2(
        splits={
            "default": TrainTestIndices(
                training=IndexSet(indices=train_indices), test=IndexSet(indices=test_indices)
            )
        }
    )
    benchmark = BenchmarkV2Specification(
        name="v2-benchmark-float-dtype",
        owner=test_org_owner,
        dataset=test_dataset_v2,
        split=split,
        target_cols=["A"],
        input_cols=["B"],
    )
    return benchmark


@pytest.fixture(scope="function")
def test_benchmark_v2_multiple_test_sets(test_dataset_v2, test_org_owner):
    """Fixture for BenchmarkV2 with multiple splits (cross-validation scenario)"""
    from polaris.benchmark._split_v2 import TrainTestIndices

    splits = {
        "split_1": TrainTestIndices(
            training=IndexSet(indices=[0, 1, 2, 3, 4]), test=IndexSet(indices=[5, 6, 7])
        ),
        "split_2": TrainTestIndices(
            training=IndexSet(indices=[0, 1, 2, 5, 6, 7]), test=IndexSet(indices=[8, 9])
        ),
        "split_3": TrainTestIndices(
            training=IndexSet(indices=[0, 1, 2, 8, 9]), test=IndexSet(indices=[10, 11, 12, 13])
        ),
    }
    split = SplitV2(splits=splits)
    benchmark = BenchmarkV2Specification(
        name="v2-benchmark-multiple-splits",
        owner=test_org_owner,
        dataset=test_dataset_v2,
        split=split,
        target_cols=["A"],
        input_cols=["B"],
    )
    return benchmark


@pytest.fixture(scope="function")
def v2_benchmark_with_rdkit_object_dtype(tmp_path, test_org_owner):
    from polaris.utils.zarr.codecs import RDKitMolCodec
    from polaris.benchmark._split_v2 import TrainTestIndices

    zarr_path = tmp_path / "test_rdkit_object_dtype.zarr"
    root = zarr.open(str(zarr_path), mode="w")
    root.array(
        "expt",
        data=np.empty(2, dtype=object),
        dtype=object,
        object_codec=RDKitMolCodec(),
    )
    root.array("smiles", data=np.array(["CCO", "CCC"]))  # Add smiles column
    zarr.consolidate_metadata(root.store)
    dataset = DatasetV2(
        name="test-dataset-v2-rdkit-object",
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
    split = SplitV2(
        splits={
            "default": TrainTestIndices(
                training=IndexSet(indices=train_indices), test=IndexSet(indices=test_indices)
            )
        }
    )
    benchmark = BenchmarkV2Specification(
        name="v2-benchmark-rdkit-object-dtype",
        owner=test_org_owner,
        dataset=dataset,
        split=split,
        target_cols=["expt"],
        input_cols=["smiles"],
    )
    return benchmark


@pytest.fixture(scope="function")
def v2_benchmark_with_atomarray_object_dtype(tmp_path, test_org_owner):
    from polaris.utils.zarr.codecs import AtomArrayCodec
    from polaris.benchmark._split_v2 import TrainTestIndices

    zarr_path = tmp_path / "test_atomarray_object_dtype.zarr"
    root = zarr.open(str(zarr_path), mode="w")
    root.array(
        "expt",
        data=np.empty(2, dtype=object),
        dtype=object,
        object_codec=AtomArrayCodec(),
    )
    root.array("smiles", data=np.array(["CCO", "CCC"]))  # Add smiles column
    zarr.consolidate_metadata(root.store)
    dataset = DatasetV2(
        name="test-dataset-v2-atomarray-object",
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
    split = SplitV2(
        splits={
            "default": TrainTestIndices(
                training=IndexSet(indices=train_indices), test=IndexSet(indices=test_indices)
            )
        }
    )
    benchmark = BenchmarkV2Specification(
        name="v2-benchmark-atomarray-object-dtype",
        owner=test_org_owner,
        dataset=dataset,
        split=split,
        target_cols=["expt"],
        input_cols=["smiles"],
    )
    return benchmark
