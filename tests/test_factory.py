import datamol as dm
import pandas as pd
import pytest
from fastpdb import struc
from zarr.errors import ContainsArrayError

from polaris.dataset import DatasetFactory, create_dataset_from_file
from polaris.dataset._factory import create_dataset_from_files
from polaris.dataset.converters import PDBConverter, SDFConverter, ZarrConverter


def _check_pdb_dataset(dataset, ground_truth):
    assert len(dataset) == len(ground_truth)
    for i in range(dataset.table.shape[0]):
        pdb_array = dataset.get_data(row=i, col="pdb")
        assert isinstance(pdb_array, struc.AtomArray)
        assert pdb_array[0] == ground_truth[i][0]
        assert pdb_array.equal_annotations(ground_truth[i])


def _check_dataset(dataset, ground_truth, mol_props_as_col):
    assert len(dataset) == len(ground_truth)

    for row in range(len(dataset)):
        mol = dataset.get_data(row=row, col="molecule")

        assert isinstance(mol, dm.Mol)

        if mol_props_as_col:
            assert not mol.HasProp("my_property")
            v = dataset.get_data(row=row, col="my_property")
            assert v == ground_truth[row].GetProp("my_property")

        else:
            assert mol.HasProp("my_property")
            assert mol.GetProp("my_property") == ground_truth[row].GetProp("my_property")
            assert "my_property" not in dataset.columns


def test_sdf_zarr_conversion(sdf_file, caffeine, tmp_path):
    """Test conversion between SDF and Zarr with utility function"""
    dataset = create_dataset_from_file(sdf_file, str(tmp_path / "archive.zarr"))
    _check_dataset(dataset, [caffeine], True)


@pytest.mark.parametrize("mol_props_as_col", [True, False])
def test_factory_sdf_with_prop_as_col(sdf_file, caffeine, tmp_path, mol_props_as_col):
    """Test conversion between SDF and Zarr with factory pattern"""

    factory = DatasetFactory(str(tmp_path / "archive.zarr"))

    converter = SDFConverter(mol_prop_as_cols=mol_props_as_col)
    factory.register_converter("sdf", converter)

    factory.add_from_file(sdf_file)
    dataset = factory.build()

    _check_dataset(dataset, [caffeine], mol_props_as_col)


def test_zarr_to_zarr_conversion(zarr_archive, tmp_path):
    """Test conversion between Zarr and Zarr with utility function"""
    dataset = create_dataset_from_file(zarr_archive, str(tmp_path / "archive.zarr"))
    assert len(dataset) == 100
    assert len(dataset.columns) == 2
    assert all(c in dataset.columns for c in ["A", "B"])
    assert all(dataset.annotations[c].is_pointer for c in ["A", "B"])
    assert dataset.get_data(row=0, col="A").shape == (2048,)


def test_zarr_with_factory_pattern(zarr_archive, tmp_path):
    """Test conversion between Zarr and Zarr with factory pattern"""

    factory = DatasetFactory(str(tmp_path / "archive.zarr"))
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


def test_factory_pdb(pdbs_structs, pdb_paths, tmp_path):
    """Test conversion between PDB file and Zarr with factory pattern"""
    factory = DatasetFactory(str(tmp_path / "pdb.zarr"))

    converter = PDBConverter()
    factory.register_converter("pdb", converter)

    factory.add_from_file(pdb_paths[0])
    dataset = factory.build()

    _check_pdb_dataset(dataset, pdbs_structs[:1])


def test_factory_pdbs(pdbs_structs, pdb_paths, tmp_path):
    """Test conversion between PDB files and Zarr with factory pattern"""

    factory = DatasetFactory(str(tmp_path / "pdbs.zarr"))

    converter = PDBConverter()
    factory.register_converter("pdb", converter)

    factory.add_from_files(pdb_paths, axis=0)
    dataset = factory.build()

    assert dataset.table.shape[0] == len(pdb_paths)
    _check_pdb_dataset(dataset, pdbs_structs)


def test_pdbs_zarr_conversion(pdbs_structs, pdb_paths, tmp_path):
    """Test conversion between PDBs and Zarr with utility function"""

    dataset = create_dataset_from_files(pdb_paths, str(tmp_path / "pdbs_2.zarr"), axis=0)

    assert dataset.table.shape[0] == len(pdb_paths)
    _check_pdb_dataset(dataset, pdbs_structs)


def test_factory_sdfs(sdf_files, caffeine, ibuprofen, tmp_path):
    """Test conversion between SDF and Zarr with factory pattern"""

    factory = DatasetFactory(str(tmp_path / "sdfs.zarr"))

    converter = SDFConverter(mol_prop_as_cols=True)
    factory.register_converter("sdf", converter)

    factory.add_from_files(sdf_files, axis=0)
    dataset = factory.build()

    _check_dataset(dataset, [caffeine, ibuprofen], True)


def test_factory_sdf_pdb(sdf_file, pdb_paths, caffeine, pdbs_structs, tmp_path):
    """Test conversion between SDF and PDB from files to Zarr with factory pattern"""

    factory = DatasetFactory(str(tmp_path / "sdf_pdb.zarr"))

    sdf_converter = SDFConverter(mol_prop_as_cols=False)
    factory.register_converter("sdf", sdf_converter)

    pdb_converter = PDBConverter()
    factory.register_converter("pdb", pdb_converter)

    factory.add_from_files([sdf_file, pdb_paths[0]], axis=1)
    dataset = factory.build()

    _check_dataset(dataset, [caffeine], False)
    _check_pdb_dataset(dataset, pdbs_structs[:1])


def test_factory_from_files_same_column(sdf_files, pdb_paths, tmp_path):
    factory = DatasetFactory(str(tmp_path / "files.zarr"))

    sdf_converter = SDFConverter(mol_prop_as_cols=False)
    factory.register_converter("sdf", sdf_converter)

    pdb_converter = PDBConverter()
    factory.register_converter("pdb", pdb_converter)

    # do not allow same type of files to be appended in columns by `add_from_files`
    # in this case, user should define converter for individual columns

    # attempt to append columns by pdbs
    with pytest.raises(ValueError):
        factory.add_from_files(pdb_paths, axis=1)

    # attempt to append columns by sdfs
    with pytest.raises(ContainsArrayError):
        factory.add_from_files(sdf_files, axis=1)
