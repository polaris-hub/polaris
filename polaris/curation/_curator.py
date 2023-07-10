# class to perform data curation for both chemistry and endpoint measured values
from typing import Union, Optional
import pandas as pd
from ._chemistry_curator import run_chemistry_curation, UNIQUE_ID, NO_STEREO_UNIQUE_ID, SMILES_COL
from ._data_curator import run_data_curation

ORI_PREFIX = "ORI_"


class MolecularCurator:
    """
    Apply curation process on given dataset on both chemistry and endpoint measurements

    Args:
        data: Data frame which contains molecule and bioactivity data columns
        mol_col: Molecule column name
        data_cols: Bioactivity data column names
        mask_stereo_undefined_mols: If set to 'True', for the molecule which contains undefined stereochemistry
            center, the bioactivity data will be set to 'Nan' and be ignored.
        ignore_stereo: If set to 'True', the stereochemistry information will be ignored from data processing steps,
            such as detect replicated molecules, detection of activity cliff.
        class_thresholds: Dictionary of bioactivity column names and the thresholds for discretizing
            the continuous data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mol_col: str,
        data_cols: str,
        mask_stereo_undefined_mols: bool = False,
        ignore_stereo: bool = False,
        class_thresholds: dict = None,
    ):
        self.data = data
        self.mol_col = mol_col
        self.data_cols = data_cols
        self.mask_stereo_undefined_mols = mask_stereo_undefined_mols
        self.ignore_stereo = ignore_stereo
        self.class_thresholds = class_thresholds

    def run(self):
        # copy the original data columns
        data = self.data.copy()
        for data_col in self.data_cols + [self.mol_col]:
            data[f"{ORI_PREFIX}{data_col}"] = self.data[data_col].copy()

        # run chemistry curation
        dataframe = run_chemistry_curation(self.data[self.mol_col], ignore_stereo=self.ignore_stereo)

        for col in dataframe.columns:
            data[col] = dataframe[col].values

        # run endpoint curation
        data = run_data_curation(
            data=data,
            data_cols=self.data_cols,
            mask_stereo_undefined_mols=self.mask_stereo_undefined_mols,
            ignore_stereo=self.ignore_stereo,
            class_thresholds=self.class_thresholds,
        )

        data.reset_index(drop=True, inplace=True)
        return data

    def __call__(self):
        return self.run()
