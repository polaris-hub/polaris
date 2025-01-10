import datamol as dm
import zarr

from polaris.dataset.zarr.codecs import RDKitMolCodec


def test_rdkit_mol_codec():
    mol = dm.to_mol("C1=CC=CC=C1")
    arr = zarr.array([mol, mol], chunks=(2,), dtype=object, object_codec=RDKitMolCodec())
    assert dm.same_mol(arr[0], mol)
    assert dm.same_mol(arr[1], mol)
