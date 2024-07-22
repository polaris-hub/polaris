import uuid
from typing import TYPE_CHECKING, Optional, Sequence, Union
from loguru import logger
import io
from tqdm import tqdm

import numpy as np
import pandas as pd
from rdkit import Chem
import datamol as dm

from polaris.dataset import ColumnAnnotation, Modality
from polaris.dataset._adapters import Adapter
from polaris.dataset.converters._base import Converter, FactoryProduct

if TYPE_CHECKING:
    from polaris.dataset import DatasetFactory


class SDFConverter(Converter):
    """
    Converts a SDF file into a Polaris dataset.

    Info: Binary strings for serialization
        This class converts the molecules to binary strings (for ML purposes, this should be lossless).
        This might not be the most storage efficient, but is fastest and easiest to maintain.
        See this [Github Discussion](https://github.com/rdkit/rdkit/discussions/7235) for more info.

    Properties defined on the molecule level in the SDF file can be extracted into separate columns
    or can be kept in the molecule object.

    Args:
        mol_column: The name of the column that will contain the pointers to the molecules.
        smiles_column: The name of the column that will contain the SMILES strings.
        use_isomeric_smiles: Whether to use isomeric SMILES.
        mol_id_column: The name of the column that will contain the molecule names.
        mol_prop_as_cols: Whether to extract properties defined on the molecule level in the SDF file into separate columns.
        groupby_key: The name of the column to group by. If set, the dataset can combine multiple pointers
            to the molecules into a single datapoint.
    """

    def __init__(
        self,
        mol_column: str = "molecule",
        smiles_column: Optional[str] = "smiles",
        use_isomeric_smiles: bool = True,
        mol_id_column: Optional[str] = None,
        mol_prop_as_cols: bool = True,
        groupby_key: Optional[str] = None,
        n_jobs: int = 1,
        zarr_chunks: Sequence[Optional[int]] = (1,),
        split: Optional[bool] = False,
        max_num_mols: Optional[int] = None,
        batch_size: Optional[int] = 128,
    ) -> None:
        super().__init__()
        self.mol_column = mol_column
        self.smiles_column = smiles_column
        self.use_isomeric_smiles = use_isomeric_smiles
        self.mol_id_column = mol_id_column
        self.mol_prop_as_cols = mol_prop_as_cols
        self.groupby_key = groupby_key
        self.n_jobs = n_jobs
        self.chunks = zarr_chunks
        self.split = split
        self.max_num_mols = max_num_mols
        self.batch_size = batch_size

    def _read_sdf_string(self, sdf_str, tmp_col):
        return dm.read_sdf(
            io.BytesIO(sdf_str.strip().encode()),
            as_df=self.mol_prop_as_cols,
            smiles_column=self.smiles_column,
            mol_column=tmp_col,
            remove_hs=False,
            sanitize=False,
        )

    def convert(self, path: str, factory: "DatasetFactory") -> FactoryProduct:
        tmp_col = uuid.uuid4().hex

        if self.split:
            with open(path) as f:
                sdf_string = f.read().strip()
            sdf_list = sdf_string.split("$$$$")
            if self.max_num_mols:
                sdf_list = sdf_list[: self.max_num_mols]

            logger.info(f"Number of SDFs: {len(sdf_list)}")

            sdf_size = len(sdf_list)
            num_batches = max(sdf_size // self.batch_size, 1)
            batches = np.array_split(sdf_list, num_batches)

            df_list = []

            index_list = np.array_split(range(sdf_size), num_batches)
            for i, sdf_str_list in enumerate(batches):
                logger.info(f"Batch {i} ...")
                df_batch = self.convert_in_batches(
                    path_or_sdf=sdf_str_list,
                    factory=factory,
                    tmp_col=tmp_col,
                    init=True if i == 0 else False,
                    pointer_index=index_list[i],
                )
                df_list.append(df_batch)
            df = pd.concat(df_list, ignore_index=True)
        else:
            df = self.convert_in_batches(path_or_sdf=path, factory=factory, tmp_col=tmp_col, init=True)

        # Set the annotations
        annotations = {self.mol_column: ColumnAnnotation(is_pointer=True, modality=Modality.MOLECULE_3D)}
        if self.smiles_column is not None:
            annotations[self.smiles_column] = ColumnAnnotation(modality=Modality.MOLECULE)

        return df, annotations, {self.mol_column: Adapter.BYTES_TO_MOL}

    def convert_in_batches(
        self,
        path_or_sdf,
        factory: "DatasetFactory",
        tmp_col,
        init: bool = False,
        pointer_index: Optional[Union[np.ndarray, list]] = None,
    ):
        if isinstance(path_or_sdf, np.ndarray):
            df = dm.parallelized(
                fn=self._read_sdf_string,
                arg_type="args",
                inputs_list=[(i, j) for i, j in zip(path_or_sdf, [tmp_col] * len(path_or_sdf))],
                n_jobs=self.n_jobs,
                scheduler="threads",
            )

            if isinstance(df[0], pd.DataFrame):
                df = pd.concat(df, ignore_index=True)
        else:
            if self.split and isinstance(path_or_sdf, str):
                path_or_sdf = io.BytesIO(path_or_sdf.strip().encode())

            df = dm.read_sdf(
                path_or_sdf,
                as_df=self.mol_prop_as_cols,
                smiles_column=self.smiles_column,
                mol_column=tmp_col,
                remove_hs=False,
                sanitize=False,
                max_num_mols=self.max_num_mols,
            )

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame({tmp_col: df})

        if self.mol_column in df.columns:
            raise ValueError(
                f"The column name '{self.mol_column}' clashes with the name of a property in the SDF file. "
                f"Please choose another name by setting the `mol_column` in the {self.__class__.__name__}."
            )

        # Add a column with the molecule name if it doesn't exist yet
        if self.mol_id_column is not None and self.mol_id_column not in df.columns:

            def _get_name(mol: dm.Mol):
                return mol.GetProp(self.mol_id_column) if mol.HasProp(self.mol_id_column) else None

            names = dm.parallelized(_get_name, df[tmp_col], n_jobs=self.n_jobs, scheduler="threads")
            df[self.mol_id_column] = names

        # Add a column with the SMILES if it doesn't exist yet
        if self.smiles_column is not None and self.smiles_column not in df.columns:
            names = dm.parallelized(
                lambda mol: dm.to_smiles(mol, isomeric=self.use_isomeric_smiles),
                df[tmp_col],
                n_jobs=self.n_jobs,
            )
            df[self.smiles_column] = names

        # Convert the molecules to binary strings. This should be lossless and efficient.
        # NOTE (cwognum): We might want to not always store private and computed properties.
        props = Chem.PropertyPickleOptions.AllProps
        if self.mol_prop_as_cols:
            props &= ~Chem.PropertyPickleOptions.MolProps
        bytes_data = np.array([mol.ToBinary(props) for mol in df[tmp_col]])

        df.drop(columns=[tmp_col], inplace=True)

        if init:
            # Create the zarr array
            factory.zarr_root.array(self.mol_column, bytes_data, dtype=bytes, chunks=self.chunks)
        else:
            # Append the zarr array
            dataset = factory.zarr_root_append[self.mol_column]
            dataset.resize((dataset.shape[0] + len(bytes_data),))
            dataset[-len(bytes_data) :] = bytes_data

        # Add a pointer column to the table
        # We support grouping by a key, to allow inputs of variable length
        grouped = pd.DataFrame(columns=[*df.columns, self.mol_column])
        if self.groupby_key is not None:
            for _, group in df.reset_index(drop=True).groupby(by=self.groupby_key):
                start = group.index[0]
                end = group.index[-1]

                if group.nunique().sum() != len(group.columns):
                    raise ValueError(
                        f"After grouping by {self.groupby_key}, values for other columns are not unique within a group. "
                        "Please handle this manually to ensure aggregation is done correctly."
                    )

                # Get the pointer path
                pointer_idx = f"{start}:{end}" if start != end else f"{start}"
                pointer = self.get_pointer(self.mol_column, pointer_idx)

                # Get the single unique value per column for the group and append
                unique_values = [group[col].unique()[0] for col in df.columns]
                grouped.loc[len(grouped)] = [*unique_values, pointer]

            df = grouped

        else:
            if pointer_index is None:
                pointer_index = range(len(df))
            pointers = [self.get_pointer(self.mol_column, i) for i in pointer_index]
            df[self.mol_column] = pd.Series(pointers)

        return df
