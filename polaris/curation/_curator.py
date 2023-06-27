# class to perform data curation for both chemistry and endpoint measured values
from typing import Union, Optional
import pandas as pd
from ._chemistry_curator import run_chemistry_curation, UNIQUE_ID, NO_STEREO_UNIQUE_ID, SMILES_COL
from ._data_curator import run_data_curation
from .utils.common import copy_columns


class Curator:
    """
    Apply curation process on given dataset on both chemistry and endpoint measurements
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mol_col: str,
        data_cols: str,
        mask_stereo_cliff: bool = False,
        ignore_stereo: bool = False,
        class_thresholds: dict = None,
    ):
        self.data = data
        self.mol_col = mol_col
        self.data_cols = data_cols
        self.mask_stereo_cliff = mask_stereo_cliff
        self.ignore_stereo = ignore_stereo
        self.class_thresholds = class_thresholds

    def run(self):
        # copy the original data
        data = copy_columns(data=self.data, data_cols=[self.mol_col] + self.data_cols)

        # run chemistry curation
        dataframe = run_chemistry_curation(data[self.mol_col], ignore_stereo=self.ignore_stereo)

        data = pd.concat([data[self.data_cols], dataframe], axis=1, ignore_index=True)
        data.columns = self.data_cols + dataframe.columns.tolist()

        # run endpoint curation
        merge_on = NO_STEREO_UNIQUE_ID if self.ignore_stereo else UNIQUE_ID
        dataframe = run_data_curation(
            data=data,
            data_cols=self.data_cols,
            mask_stereo_cliff=self.mask_stereo_cliff,
            ignore_stereo=self.ignore_stereo,
            class_thresholds=self.class_thresholds,
        )
        data = dataframe.merge(on=merge_on, right=data[[merge_on, SMILES_COL]])

        data.reset_index(drop=True, inplace=True)
        return data
