from typing import TYPE_CHECKING, Optional, Sequence, List, Union

from pathlib import Path
import numpy as np
import pandas as pd
import fastpdb

from polaris.dataset import ColumnAnnotation, Modality
from polaris.dataset._adapters import Adapter
from polaris.dataset.converters._base import Converter, FactoryProduct

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


class PDBConverter(Converter):
    """
    Converts PDB files into a Polaris dataset.

    Info: Numpy array for serialization
        This class converts the 3D structure to fastpdb array (for ML purposes with key structual information).
        This might not be the most storage efficient, but is fastest and easiest to maintain.
        See[fastpdb](https://github.com/biotite-dev/fastpdb) and [biotite](https://github.com/biotite-dev/biotite/blob/main/src/biotite/structure/atoms.py) for more info.

    Args:
        pdb_column: The name of the column that will contain the pointers to the pdbs.
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

    def _load_pdf(self, path: str, pdb_pointer=None) -> dict:
        """
        Load a single PDB file along with a dictionary that contains structure properties as keys.

        `AtomArray` stores data for an entire structure model containing *n* atoms.
        See [AtomArray](https://github.com/biotite-dev/biotite/blob/0404084283765baef765cc869d86ee07a898d82f/src/biotite/structure/atoms.py#L558) for more details.
        """
        # load pdb file
        in_file = fastpdb.PDBFile.read(path)

        # get the structure as AtomArray
        atom_array = in_file.get_structure(
            model=1, include_bonds=True, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
        )

        # convert AtomArray to dict
        pdb_dict = self._pdb_to_dict(atom_array)

        return pdb_dict

    def _convert_pdb(
        self, path: str, factory: "DatasetFactory", pdb_pointer: Union[str, int]
    ) -> FactoryProduct:
        """
        Convert a single pdb to zarr file
        """

        # load pdb as fastpbd and convert to dict
        pdb_dict = self._load_pdf(path)

        # Create group and add datasets
        if self.pdb_column not in factory.zarr_root:
            pdb_group = factory.zarr_root.create_group(self.pdb_column)
        else:
            pdb_group = factory.zarr_root[self.pdb_column]

        group = pdb_group.create_group(pdb_pointer)
        for col_name, col_val in pdb_dict.items():
            col_val = np.array(col_val)
            # get the accepted dtype
            dtype = DTYPE_DICT.get(col_name, col_val.dtype)
            group.create_dataset(col_name, data=col_val, dtype=dtype)

    def convert(self, path: Union[str, List[str]], factory: "DatasetFactory") -> FactoryProduct:
        """Convert one or a list of PDB files into Zarr"""

        if not isinstance(path, list):
            path = [path]

        pdb_pointers = []

        # load single pdb and convert to zarr
        for pdb_path in path:
            # use the pdb file name as pointer
            pdb_pointer = Path(pdb_path).stem
            self._convert_pdb(pdb_path, factory, pdb_pointer)
            pdb_pointers.append(pdb_pointer)

        # Add a pointer column to the table
        pointers = [self.get_pointer(self.pdb_column, pointer) for pointer in pdb_pointers]
        df = pd.DataFrame()
        df[self.pdb_column] = pd.Series(pointers)

        # Set the annotations
        annotations = {self.pdb_column: ColumnAnnotation(is_pointer=True, modality=Modality.PROTEIN_3D)}

        # Return the dataframe and the annotations
        return df, annotations, {self.pdb_column: Adapter.PDB_TO_ARRAY}
