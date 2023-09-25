# class to perform data curation for both chemistry and endpoint measured values
from typing import List, Optional

from pydantic import BaseModel

from ._chemistry_curator import run_chemistry_curation
from ._data_curator import run_data_curation
from .utils import PandasDataFrame

ORI_PREFIX = "ORIGINAL_"


class MolecularCurator(BaseModel):
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
        outlier_params: Parameters for automatic outlier detection.
        progress: Whether show progress of the parallelization process.

    See Also:
        <polaris.curation._data_curator.run_data_curation>
        <polaris.curation._chemistry_curator.run_chemistry_curation>
    """

    data: PandasDataFrame
    mol_col: str = "smiles"
    data_cols: List[str]
    mask_stereo_undefined_mols: bool = False
    ignore_stereo: bool = False
    class_thresholds: Optional[dict] = None
    outlier_params: Optional[dict] = None
    activity_cliff_params: Optional[dict] = None
    progress: bool = False

    def __call__(self):
        # copy the original data columns
        data = self.data.copy()
        for data_col in self.data_cols + [self.mol_col]:
            data[f"{ORI_PREFIX}{data_col}"] = self.data[data_col].copy()

        # run chemistry curation
        dataframe = run_chemistry_curation(
            self.data[self.mol_col], ignore_stereo=self.ignore_stereo, progress=False
        )

        for col in dataframe.columns:
            data[col] = dataframe[col].values

        # run endpoint curation
        data = run_data_curation(
            data=data,
            data_cols=self.data_cols,
            mask_stereo_undefined_mols=self.mask_stereo_undefined_mols,
            ignore_stereo=self.ignore_stereo,
            class_thresholds=self.class_thresholds,
            outlier_params=self.outlier_params,
            activity_cliff_params=self.activity_cliff_params,
        )

        data = data.reset_index(drop=True)
        return data
