from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import fastpdb
import numpy as np
import pandas as pd
import zarr
from fastpdb import struc

from polaris.dataset import ColumnAnnotation, Modality
from polaris.dataset._adapters import Adapter
from polaris.dataset.converters._base import Converter, FactoryProduct
from polaris.dataset.zarr._utils import load_zarr_group_to_memory

if TYPE_CHECKING:
    from polaris.dataset import DatasetFactory


DTYPE_DICT = {
    "chain_id": "U4",
    "res_id": int,
    "ins_code": "U1",
    "res_name": "U5",
    "hetero": bool,
    "atom_name": "U6",
    "element": "U2",
}

KEYS = [
    "X",
    "Y",
    "Z",
    "chain_id",
    "res_id",
    "ins_code",
    "res_name",
    "hetero",
    "atom_name",
    "element",
    "atom_id",
    "b_factor",
    "occupancy",
    "charge",
]


def zarr_to_pdb(atom_dict: zarr.Group):
    """Load a dictionary of arrays to fastpdb AtomArray"""

    atom_array = []

    # Load to memory once to drastically speed up the conversion
    # Otherwise you keep decompressing and copying entire chunks of data to just use a single row
    atom_dict = load_zarr_group_to_memory(atom_dict)

    # convert dictionary to array list of Atom object
    array_length = atom_dict["X"].shape[0]
    for ind in range(array_length):
        atom = struc.Atom(
            coord=(atom_dict["X"][ind], atom_dict["Y"][ind], atom_dict["Z"][ind]),
            chain_id=atom_dict["chain_id"][ind],
            res_id=atom_dict["res_id"][ind],
            ins_code=atom_dict["ins_code"][ind],
            res_name=atom_dict["res_name"][ind],
            hetero=atom_dict["hetero"][ind],
            atom_name=atom_dict["atom_name"][ind],
            element=atom_dict["element"][ind],
            b_factor=atom_dict["b_factor"][ind],
            occupancy=atom_dict["occupancy"][ind],
            charge=atom_dict["charge"][ind],
            atom_id=atom_dict["atom_id"][ind],
        )
        atom_array.append(atom)

    # Note that this is a `fastpdb` AtomArray, not a NumPy array.
    return struc.array(atom_array)


class PDBConverter(Converter):
    """
    Converts PDB files into a Polaris dataset based on fastpdb.

    Info: Only the most essential structural information of a protein is retained
        This conversion saves the 3D coordinates, chain ID, residue ID, insertion code, residue name, heteroatom indicator, atom name, element, atom ID, B-factor, occupancy, and charge.
        Records such as CONECT (connectivity information), ANISOU (anisotropic Temperature Factors), HETATM (heteroatoms and ligands) are handled by `fastpdb`.
        We believe this makes for a good _ML-ready_ format, but let us know if you require any other information to be saved.


    Info: PDBs as ND-arrays using `biotite`
        To save PDBs in a Polaris-compatible format, we convert them to ND-arrays using `fastpdb` and `biotite`.
        We then save these ND-arrays to Zarr archives.
        For more info, see [fastpdb](https://github.com/biotite-dev/fastpdb)
        and [biotite](https://github.com/biotite-dev/biotite/blob/main/src/biotite/structure/atoms.py)

    Args:
        pdb_column: The name of the column that will contain the pointers to the pdbs.
        n_jobs: The number of jobs to run in parallel.
        zarr_chunks: The chunk size for the Zarr arrays.
    """

    def __init__(
        self,
        pdb_column: str = "pdb",
        n_jobs: int = 1,
        zarr_chunks: Sequence[Optional[int]] = (1,),
    ) -> None:
        super().__init__()
        self.pdb_column = pdb_column
        self.n_jobs = n_jobs
        self.chunks = zarr_chunks

    def _pdb_to_dict(self, atom_array: np.ndarray):
        """
        Convert 'biotite.AtomArray' to dictionary
        """
        pdb_dict = {k: [] for k in KEYS}
        for row in atom_array:
            pdb_dict["X"].append(row.coord[0])
            pdb_dict["Y"].append(row.coord[1])
            pdb_dict["Z"].append(row.coord[2])
            pdb_dict["chain_id"].append(row.chain_id)
            pdb_dict["res_id"].append(row.res_id)
            pdb_dict["ins_code"].append(row.ins_code)
            pdb_dict["res_name"].append(row.res_name)
            pdb_dict["hetero"].append(row.hetero)
            pdb_dict["atom_name"].append(row.atom_name)
            pdb_dict["element"].append(row.element)
            pdb_dict["atom_id"].append(row.atom_id)
            pdb_dict["b_factor"].append(row.b_factor)
            pdb_dict["occupancy"].append(row.occupancy)
            pdb_dict["charge"].append(row.charge)
        return pdb_dict

    def _load_pdb(self, path: str, pdb_pointer=None) -> dict:
        """
        Load a single PDB file along with a dictionary that contains structure properties as keys.

        `AtomArray` stores data for an entire structure model containing *n* atoms.
        See [AtomArray](https://github.com/biotite-dev/biotite/blob/0404084283765baef765cc869d86ee07a898d82f/src/biotite/structure/atoms.py#L558) for more details.
        """
        # load pdb file
        in_file = fastpdb.PDBFile.read(path)

        # get the structure as AtomArray
        atom_array = in_file.get_structure(
            model=1,
            include_bonds=True,
            extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
        )

        # convert AtomArray to dict
        pdb_dict = self._pdb_to_dict(atom_array)

        return pdb_dict

    def _convert_pdb(
        self, path: str, factory: "DatasetFactory", pdb_pointer: Union[str, int], append: bool = False
    ) -> FactoryProduct:
        """
        Convert a single pdb to zarr file
        """

        # load pdb as fastpbd and convert to dict
        pdb_dict = self._load_pdb(path)

        # Create group and add
        if append:
            if self.pdb_column not in factory.zarr_root:
                raise RuntimeError(
                    f"Group {self.pdb_column} doesn't exist in {factory.zarr_root}. \
                    Please make sure the Group {self.pdb_column} is created. Or set `append` to `False`."
                )
            else:
                pdb_group = factory.zarr_root[self.pdb_column]
        else:
            pdb_group = factory.zarr_root.create_group(self.pdb_column)

        group = pdb_group.create_group(pdb_pointer)
        for col_name, col_val in pdb_dict.items():
            col_val = np.array(col_val)
            # get the accepted dtype
            dtype = DTYPE_DICT.get(col_name, col_val.dtype)
            group.array(col_name, data=col_val, dtype=dtype)

    def convert(self, path, factory: "DatasetFactory", append: bool = False) -> FactoryProduct:
        """Convert one or a list of PDB files into Zarr"""
        pdb_pointers = []

        # load single pdb and convert to zarr
        pdb_pointer = Path(path).stem
        self._convert_pdb(path, factory, pdb_pointer, append)
        pdb_pointers.append(pdb_pointer)

        # Add a pointer column (path) to the table
        pointers = [f"{self.pdb_column}/{pointer}" for pointer in pdb_pointers]
        df = pd.DataFrame()
        df[self.pdb_column] = pd.Series(pointers)

        # Set the annotations
        annotations = {
            self.pdb_column: ColumnAnnotation(
                is_pointer=True, modality=Modality.PROTEIN_3D, content_type="chemical/x-pdb"
            )
        }

        # Return the dataframe and the annotations
        return df, annotations, {self.pdb_column: Adapter.ARRAY_TO_PDB}
