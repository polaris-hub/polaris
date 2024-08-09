import uuid
from typing import TYPE_CHECKING, Optional, Sequence, List

import numpy as np
import datamol as dm
import pandas as pd
from rdkit import Chem
import zarr

import biotite.structure as struc
from biotite.structure import Atom
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
    """ """

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

    def _load_pdf(self, path: str):
        """Load PDB file"""
        # load structure
        in_file = fastpdb.PDBFile.read(pdb_file)
        if pdb_pointer is None:
            pdb_pointer = Path(pdb_file).stem
        atom_array = in_file.get_structure(
            model=1, include_bonds=True, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
        )

        # fastpbd to dict
        pdb_dict = pdb_to_dict(atom_array)

        return pdb_dict

    def _convert_pdb(self, path: str, factory: "DatasetFactory") -> FactoryProduct:
        # load structure
        in_file = fastpdb.PDBFile.read(pdb_file)
        if pdb_pointer is None:
            pdb_pointer = Path(pdb_file).stem
        atom_array = in_file.get_structure(
            model=1, include_bonds=True, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
        )

        # fastpbd to dict
        pdb_dict = pdb_to_dict(atom_array)

        # Create the zarr array
        # factory.zarr_root.array(self.pdb_column, bytes_data, dtype=bytes, chunks=self.chunks)

        # Create a Zarr store
        store = zarr.DirectoryStore(zarr_file)

        # Create a root group
        if mode == "a":
            root = zarr.open_group(store=store, mode=mode)
        else:
            root = zarr.group(store=store)

        # Create group and add datasets
        pdb_group = root.create_group("pdb")
        group = pdb_group.create_group(pdb_pointer)
        for col_name, col_val in pdb_dict.items():
            col_val = np.array(col_val)
            # get the accepted dtype
            dtype = DTYPE_DICT.get(col_name, col_val.dtype)
            group.create_dataset(col_name, data=col_val, dtype=dtype)

    def convert(self, path_list: List[str], factory: "DatasetFactory") -> FactoryProduct:
        
        
        # load single pdb and convert to zarr
        for path in path_list:
            self._convert_pdb(path, factory=factory)

        # Add a pointer column to the table
        pointers = [self.get_pointer(self.pdb_column, i) for i in range(len(path_list))]
        df[self.pdb_column] = pd.Series(pointers)

        # Set the annotations
        annotations = {self.pdb_column: ColumnAnnotation(is_pointer=True, modality=Modality.PROTEIN_3D)}

        # Return the dataframe and the annotations
        return df, annotations, {self.pdb_column: Adapter.PDB_TO_ARRAY}
