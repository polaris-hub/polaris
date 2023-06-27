# script to clean curate endpoint measures
import numpy as np
from typing import List
from typing import Dict
import pandas as pd

from scipy.stats import zscore
from .utils.class_convertor import DatasetClassConverter
from ._chemistry_curator import UNIQUE_ID, NO_STEREO_UNIQUE_ID

DUP_COL = "is_duplicated"
MAX_CLASSES = 10
CAT = "categorical"
CNT = "continuous"


def _detect_data_type(values: np.ndarray):
    """Detect the type of input values `continuous` or `categorical`"""
    num_vals = len(np.unique(values))
    if num_vals < 2:
        return None
    elif num_vals > MAX_CLASSES:
        return CNT
    return CAT


def _detect_stereo_activity_cliff(data: pd.DataFrame, data_col: str) -> List:
    """Detect molecule stereoisomers which show activity cliff.
       The activity cliff is defined either by `zscore > 3` for continuous value,
       or different class label for categorical values.

    Args:
        data: Dataframe which contains unique ID which identifies the stereoisomers.
        data_col: Column name for the targeted activity
    """
    data_type = _detect_data_type(data[data_col].dropna())

    if data_type is None:
        raise ValueError(f"The column {data_col} contains less than 2 unique values.")

    mol_with_cliff = []
    for hashmol_id, group_df in data.groupby(NO_STEREO_UNIQUE_ID):
        if data_type == CNT:
            group_df["zscore"] = zscore(group_df[data_col])
            has_cliff = group_df["zscore"].abs().max() > 3
        else:
            has_cliff = len(group_df[data_col].unique()) > 1

        if has_cliff:
            mol_with_cliff.append(hashmol_id)

    return mol_with_cliff


def _process_stereo_activity_cliff(
    data: pd.DataFrame,
    data_cols: List[str],
    mask_stereo_cliff: bool = False,
):
    """Detect and/or mask out the molecule stereoisomers which show activity cliff

    Args:
        data: Parsed dataset
        data_cols: Data column names to be processed
        mask_stereo_cliff: Whether mask out the stereoisomers which show activity cliff from dataset.
            This is recommended if the downstream molecule representation is unable to identify the stereoisomers.
    """
    for data_col in data_cols:
        mol_with_cliff = _detect_stereo_activity_cliff(data, data_col)
        data.loc[data[NO_STEREO_UNIQUE_ID].isin(mol_with_cliff), f"{data_col}_stereo_cliff"] = True
        if mask_stereo_cliff:
            data.loc[data[NO_STEREO_UNIQUE_ID].isin(mol_with_cliff), data_col] = None
    return data


def _merge_duplicates(data: pd.DataFrame, data_cols: List[str], merge_on: str = UNIQUE_ID) -> pd.DataFrame:
    """Merge molecules with multiple measurements.
        Median is used to compute the average value cross all measurements

    Args:
        data: Parsed dataset
        data_cols: Data column names to be processed
        merge_on: Column name to identify the duplicated molecules

    """
    merged_df = []
    id_cols = list({merge_on, NO_STEREO_UNIQUE_ID})
    for uid, df in data.groupby(by=id_cols):
        uid = list(uid) if isinstance(uid, tuple) else [uid]
        merged_df.append(uid + df[data_cols].median(axis=0, skipna=True).tolist())
    merged_df = pd.DataFrame(merged_df, columns=id_cols + data_cols)
    return merged_df


def _class_conversion(
    data: pd.DataFrame,
    data_cols: List[str],
    conversion_params: List[dict],
    prefix: str = "CLASS_",
) -> pd.DataFrame:
    """
    Apply binary or multiclass conversion to the data columns by the given thresholds

    Args:
        data: Parsed dataset
        data_cols: Data column names to be converted
        conversion_params: List of parameters for class conversion
            '''
            Example:
                [
                 # for data column 1 listed in `data_cols` - binary class
                 {"threshold_values": [0.5], "threshold_labels": [0, 1],"nan_value": "NA" },
                 # for data column 2 listed in `data_cols` - multiclass
                 {"threshold_values": [10, 20], "threshold_labels": [0, 1, 2],"nan_value": "-"}
                ]

            '''
        prefix: Prefix for the added column names

    Returns:
        dataset: Dataset with clss columns
    """
    data_converter = DatasetClassConverter(
        data_columns=data_cols,
        converter_params=conversion_params,
    )
    dataset = data_converter.transform(dataset=data, prefix=prefix)
    return dataset


def run_data_curation(
    data: pd.DataFrame,
    data_cols: List[str],
    mask_stereo_cliff: bool = False,
    ignore_stereo: bool = False,
    class_thresholds: Dict = None,
) -> pd.DataFrame:
    """Perform curation on the measured values from biological assay

    Args:
        data: Dataframe which include molecule, measured values.
        data_cols: Column names for the data of interest.
        mask_stereo_cliff: Whether mask the molecule stereo pairs which show activity cliff from dataset.
        ignore_stereo: Ignore the stereochemistry information from data curation.
        class_thresholds: Parameters to define class labels for the above listed `data_cols`.
    """
    # User should deal with scaling and normalization on their end

    # merge
    data = _merge_duplicates(
        data=data, data_cols=data_cols, merge_on=NO_STEREO_UNIQUE_ID if ignore_stereo else UNIQUE_ID
    )

    # class conversion
    if class_thresholds:
        data = _class_conversion(data, data_cols, class_thresholds, prefix="CLASS_")

    if not ignore_stereo:
        # detect stereo activity cliff, keep or remove
        data = _process_stereo_activity_cliff(
            data=data,
            data_cols=[f"CLASS_{data_col}" for data_col in data_cols] if class_thresholds else data_cols,
            mask_stereo_cliff=mask_stereo_cliff,
        )

    return data
