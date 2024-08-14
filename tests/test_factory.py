import datamol as dm
import pandas as pd
import pytest

import biotite.database.rcsb as rcsb
from fastpdb import struc

from polaris.dataset import DatasetFactory, create_dataset_from_file
from polaris.dataset.converters import SDFConverter, ZarrConverter, PDBConverter


def _check_pdb_dataset(dataset, ground_truth):
    assert len(dataset) == len(ground_truth)
    for i in range(dataset.table.shape[0]):
        pdb_array = dataset.get_data(row=i, col="pdb")
        assert isinstance(pdb_array, struc.AtomArray)
        assert pdb_array[0].__eq__(ground_truth[i][0])
        assert pdb_array.equal_annotations(ground_truth[i])


def _check_dataset(dataset, ground_truth, mol_props_as_col):
    assert len(dataset) == 1

    mol = dataset.get_data(row=0, col="molecule")

    assert isinstance(mol, dm.Mol)

    if mol_props_as_col:
        assert not mol.HasProp("my_property")
        v = dataset.get_data(row=0, col="my_property")
        assert v == ground_truth.GetProp("my_property")

    else:
        assert mol.HasProp("my_property")
        assert mol.GetProp("my_property") == ground_truth.GetProp("my_property")
        assert "my_property" not in dataset.columns


def test_sdf_zarr_conversion(sdf_file, caffeine, tmpdir):
    """Test conversion between SDF and Zarr with utility function"""
    dataset = create_dataset_from_file(sdf_file, tmpdir.join("archive.zarr"))
    _check_dataset(dataset, caffeine, True)


@pytest.mark.parametrize("mol_props_as_col", [True, False])
def test_factory_sdf_with_prop_as_col(sdf_file, caffeine, tmpdir, mol_props_as_col):
    """Test conversion between SDF and Zarr with factory pattern"""

    factory = DatasetFactory(tmpdir.join("archive.zarr"))

    converter = SDFConverter(mol_prop_as_cols=mol_props_as_col)
    factory.register_converter("sdf", converter)

    factory.add_from_file(sdf_file)
    dataset = factory.build()

    _check_dataset(dataset, caffeine, mol_props_as_col)


def test_zarr_to_zarr_conversion(zarr_archive, tmpdir):
    """Test conversion between Zarr and Zarr with utility function"""
    dataset = create_dataset_from_file(zarr_archive, tmpdir.join("archive.zarr"))
    assert len(dataset) == 100
    assert len(dataset.columns) == 2
    assert all(c in dataset.columns for c in ["A", "B"])
    assert all(dataset.annotations[c].is_pointer for c in ["A", "B"])
    assert dataset.get_data(row=0, col="A").shape == (2048,)


def test_zarr_with_factory_pattern(zarr_archive, tmpdir):
    """Test conversion between Zarr and Zarr with factory pattern"""

    factory = DatasetFactory(tmpdir.join("archive.zarr"))
    converter = ZarrConverter()
    factory.register_converter("zarr", converter)
    factory.add_from_file(zarr_archive)

    factory.add_column(pd.Series([1, 2, 3, 4] * 25, name="C"))

    df = pd.DataFrame({"C": [1, 2, 3, 4], "D": ["W", "X", "Y", "Z"]})
    factory.add_columns(df, merge_on="C")

    dataset = factory.build()
    assert len(dataset) == 100
    assert len(dataset.columns) == 4
    assert all(c in dataset.columns for c in ["A", "B", "C", "D"])
    assert dataset.table["C"].apply({1: "W", 2: "X", 3: "Y", 4: "Z"}.get).equals(dataset.table["D"])


def test_factory_pdb(pdbs, tmpdir):
    """Test conversion between PDB file and Zarr with factory pattern"""
    pdb_id = "1l2y"
    pdb_path = rcsb.fetch(pdb_id, "pdb", tmpdir)

    factory = DatasetFactory(tmpdir.join("pdb.zarr"))

    converter = PDBConverter()
    factory.register_converter("pdb", converter)

    factory.add_from_file(pdb_path)
    dataset = factory.build()

    _check_pdb_dataset(dataset, pdbs[:1])


def test_factory_pdbs(pdbs, tmpdir):
    """Test conversion between PDB files and Zarr with factory pattern"""
    pdb_ids = ["1l2y", "4i23"]
    pdb_paths = rcsb.fetch(pdb_ids, "pdb", tmpdir)
    print(pdb_paths)
    factory = DatasetFactory(tmpdir.join("pdbs.zarr"))

    converter = PDBConverter()
    factory.register_converter("pdb", converter)

    factory.add_from_files(pdb_paths, axis=0)
    dataset = factory.build()

    assert dataset.table.shape[0] == len(pdb_ids)
    _check_pdb_dataset(dataset, pdbs)
