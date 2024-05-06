import datamol as dm
import numpy as np
import pytest
import zarr
from datamol.utils import fs

import polaris as po
from polaris.benchmark import (
    MultiTaskBenchmarkSpecification,
    SingleTaskBenchmarkSpecification,
)
from polaris.dataset import ColumnAnnotation, Dataset
from polaris.utils.types import HubOwner


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
    mol.SetProp("my_property", "my_value")
    return mol


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


@pytest.fixture(scope="module")
def test_dataset(test_data, test_org_owner):
    dataset = Dataset(
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
def zarr_archive(tmp_path):
    tmp_path = fs.join(tmp_path, "data.zarr")
    root = zarr.open(tmp_path, mode="w")
    root.array("A", data=np.random.random((100, 2048)))
    root.array("B", data=np.random.random((100, 2048)))
    zarr.consolidate_metadata(root.store)
    return tmp_path


@pytest.fixture(scope="function")
def test_single_task_benchmark(test_dataset):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-benchmark",
        dataset=test_dataset,
        metrics=[
            "mean_absolute_error",
            "mean_squared_error",
            "r2",
            "spearmanr",
            "pearsonr",
            "explained_var",
            "absolute_average_fold_error",
        ],
        main_metric="mean_absolute_error",
        split=(train_indices, test_indices),
        target_cols="expt",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_single_task_benchmark_clf(test_dataset):
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-benchmark",
        dataset=test_dataset,
        main_metric="accuracy",
        metrics=["accuracy", "f1", "roc_auc", "pr_auc", "mcc", "cohen_kappa", "balanced_accuracy"],
        split=(train_indices, test_indices),
        target_cols="CLASS_expt",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_single_task_benchmark_multi_clf(test_dataset):
    np.random.seed(111)
    indices = np.arange(100)
    np.random.shuffle(indices)
    train_indices = indices[:80]
    test_indices = indices[80:]

    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-benchmark",
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
def test_single_task_benchmark_multiple_test_sets(test_dataset):
    train_indices = list(range(90))
    test_indices = {"test_1": list(range(90, 95)), "test_2": list(range(95, 100))}
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-benchmark",
        dataset=test_dataset,
        metrics=[
            "mean_absolute_error",
            "mean_squared_error",
            "r2",
            "spearmanr",
            "pearsonr",
            "explained_var",
            "absolute_average_fold_error",
        ],
        main_metric="r2",
        split=(train_indices, test_indices),
        target_cols="expt",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_single_task_benchmark_clf_multiple_test_sets(test_dataset):
    np.random.seed(111)  # make sure two classes in `y_true`
    indices = np.arange(100)
    np.random.shuffle(indices)
    train_indices = indices[:80]
    test_indices = {"test_1": indices[80:90], "test_2": indices[90:]}
    benchmark = SingleTaskBenchmarkSpecification(
        name="single-task-benchmark-clf",
        dataset=test_dataset,
        metrics=["accuracy", "f1", "roc_auc", "pr_auc", "mcc", "cohen_kappa"],
        main_metric="pr_auc",
        split=(train_indices, test_indices),
        target_cols="CLASS_calc",
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_multi_task_benchmark(test_dataset):
    # For the sake of simplicity, just use a small set of indices
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    benchmark = MultiTaskBenchmarkSpecification(
        name="multi-task-benchmark",
        dataset=test_dataset,
        main_metric="mean_absolute_error",
        metrics=[
            "mean_absolute_error",
            "mean_squared_error",
            "r2",
            "spearmanr",
            "pearsonr",
            "explained_var",
            "absolute_average_fold_error",
        ],
        split=(train_indices, test_indices),
        target_cols=["expt", "calc"],
        input_cols="smiles",
        target_types={"expt": "regression"},
    )
    check_version(benchmark)
    return benchmark


@pytest.fixture(scope="function")
def test_multi_task_benchmark_clf(test_dataset):
    # For the sake of simplicity, just use a small set of indices
    train_indices = list(range(90))
    test_indices = list(range(90, 100))
    benchmark = MultiTaskBenchmarkSpecification(
        name="multi-task-benchmark",
        dataset=test_dataset,
        main_metric="accuracy",
        metrics=["accuracy", "f1", "roc_auc", "pr_auc", "mcc", "cohen_kappa"],
        split=(train_indices, test_indices),
        target_cols=["CLASS_expt", "CLASS_calc"],
        input_cols="smiles",
    )
    check_version(benchmark)
    return benchmark
