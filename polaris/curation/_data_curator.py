from typing import List
from typing import Dict
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from scipy.stats import zscore
from .utils import discretizer
from ._chemistry_curator import UNIQUE_ID, NO_STEREO_UNIQUE_ID, NUM_DEF_STEREO_CENTER, NUM_STEREO_CENTER

CAT = ["binary", "multiclass"]
CNT = ["continuous"]
CLS_PREFIX = "CLASS_"


def _detect_stereo_activity_cliff(data: pd.DataFrame, data_col: str) -> List:
    """Detect molecule stereoisomers which show activity cliff.
       For continuous data, the activity cliff is defined by the `zscore` difference greater than 1.
       For categorical data, the activity cliff is defined by class labels for categorical values.
       It's recommended to use provide clear thresholds to classify the bioactivity if possible.

    Args:
        data: Dataframe which contains unique ID which identifies the stereoisomers.
        data_col: Column name for the targeted activity
    """
    data_type = type_of_target(data[data_col].dropna().values, input_name=data_col)

    if data_type is None:
        raise ValueError(f"The column {data_col} contains less than 2 unique values.")

    mol_with_cliff = []
    if data_type in CNT:
        data[f"{data_col}_zscore"] = zscore(data[data_col].values, nan_policy="omit")

    for hashmol_id, group_df in data.groupby(NO_STEREO_UNIQUE_ID):
        if group_df.shape[0] > 1:
            if data_type in CNT:
                has_cliff = (group_df[f"{data_col}_zscore"].max() - group_df[f"{data_col}_zscore"].min()) > 1
            else:
                has_cliff = len(group_df[data_col].unique()) > 1

            if has_cliff:
                mol_with_cliff.append(hashmol_id)

    return mol_with_cliff


def _process_stereo_activity_cliff(
    data: pd.DataFrame, data_cols: List[str], mask_stereo_undefined_mols: bool = True
):
    """Detect and/or mask out the molecule stereoisomers which show activity cliff.
       Teh

    Args:
        data: Parsed dataset
        data_cols: Data column names to be processed.
        mask_stereo_undefined_mols: Whether remove molecules and the bioactivity which shows chiral shift
                                    but stereo information not defined in the molecule representation.
    """
    for data_col in data_cols:
        data_col_mask = (
            [data_col, data_col[len(CLS_PREFIX) :]] if data_col.startswith(CLS_PREFIX) else data_col
        )
        mol_with_cliff = _detect_stereo_activity_cliff(data, data_col)
        data.loc[data[NO_STEREO_UNIQUE_ID].isin(mol_with_cliff), f"{data_col}_stereo_cliff"] = True
        if len(mol_with_cliff) > 0 and mask_stereo_undefined_mols:
            mol_ids = data.query(f"`{data_col}_stereo_cliff`==True & {NUM_DEF_STEREO_CENTER}==0")[
                UNIQUE_ID
            ].values
            data.loc[data[UNIQUE_ID].isin(mol_ids), data_col_mask] = None
    return data


def _merge_duplicates(
    data: pd.DataFrame, data_cols: List[str], merge_on: List[str] = [UNIQUE_ID], keep_all_rows: bool = False
) -> pd.DataFrame:
    """Merge molecules with multiple measurements.
       To be robust to the outliers, `median` is used to compute the average value cross all measurements.

    Args:
        data: Parsed dataset
        data_cols: Data column names to be processed
        merge_on: Column name to identify the duplicated molecules.
                  dy default, `molhash_id` is used to identify the replicated molecule.

    """
    df_list = []
    # Include 'molhash without stereo information' additionally for easily tracking the stereoisomers
    for uid, df in data.groupby(by=merge_on):
        data_vals = df[data_cols].median(axis=0, skipna=True).tolist()
        df.loc[:, data_cols] = data_vals
        df_list.append(df)
    merged_df = pd.concat(df_list).sort_values(by=merge_on)
    if not keep_all_rows:
        merged_df = merged_df.drop_duplicates(subset=merge_on, keep="first").reset_index(drop=True)
    return merged_df


def _class_conversion(
    data: pd.DataFrame,
    data_cols: List[str],
    conversion_params: List[dict],
    prefix: str = CLS_PREFIX,
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
    for data_col in data_cols:
        thresholds = conversion_params.get(data_col, None)
        if thresholds is not None:
            data[f"{prefix}{data_col}"] = discretizer(X=data[data_col].values, allow_nan=True, **thresholds)
    return data


def run_data_curation(
    data: pd.DataFrame,
    data_cols: List[str],
    mask_stereo_undefined_mols: bool = False,
    ignore_stereo: bool = False,
    class_thresholds: Dict = None,
) -> pd.DataFrame:
    """Perform curation on the measured values from biological assay

    Args:
        data: Dataframe which include molecule, measured values.
        data_cols: Column names for the data of interest.
        mask_stereo_undefined_mols: Whether mask the molecule stereo pairs which show activity cliff from dataset.
        ignore_stereo: Ignore the stereochemistry information from data curation.
        class_thresholds: Parameters to define class labels for the above listed `data_cols`.
    """
    # User should deal with scaling and normalization on their end

    data = _merge_duplicates(
        data=data,
        data_cols=data_cols,
        merge_on=[NO_STEREO_UNIQUE_ID] if ignore_stereo else [UNIQUE_ID],
    )
    # class conversion
    if class_thresholds:
        data = _class_conversion(data, data_cols, class_thresholds, prefix=CLS_PREFIX)
    if not ignore_stereo:
        # detect stereo activity cliff, keep or remove
        data = _process_stereo_activity_cliff(
            data=data,
            data_cols=[
                f"{CLS_PREFIX}{data_col}" if f"{CLS_PREFIX}{data_col}" in data.columns else data_col
                for data_col in data_cols
            ]
            if class_thresholds
            else data_cols,
            mask_stereo_undefined_mols=mask_stereo_undefined_mols,
        )

    return data
