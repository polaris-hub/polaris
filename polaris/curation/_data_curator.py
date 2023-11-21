from typing import Dict, List

import pandas as pd
from loguru import logger
from sklearn.utils.multiclass import type_of_target

from ._chemistry_curator import NO_STEREO_UNIQUE_ID, UNDEF_EZ, UNDEF_ED, UNIQUE_ID
from .utils import discretizer, modified_zscore, outlier_detection

CATEGORIES = ["binary", "multiclass"]
CONTINUOUS = ["continuous"]
CLASS_PREFIX = "CLASS_"


def _identify_stereoisomers_with_activity_cliff(
    data: pd.DataFrame, data_col: str, threshold: float = 1, groupby_col: str = NO_STEREO_UNIQUE_ID
) -> List:
    """Identify the stereoisomers which show activity cliff.
       For continuous data, the activity cliff is defined by the `zscore` difference greater than 1.
       For categorical data, the activity cliff is defined by class labels for categorical values.
       It's recommended to provide clear thresholds to classify the bioactivity if possible.

    Args:
        data: Dataframe which contains unique ID which identifies the stereoisomers.
        data_col: Column name for the targeted activity
        threshold: Zscore threshold which defines the activity cliff.
                   By default, for continuous values, the activity cliff is defined as if the difference between zscores
                   is greater than 1.
        groupby_col: Column name which can be used to identifier the stereoisomers.

    Returns:
        mol_hash_ids_with_cliff: A list mol hash ids for the stereoisomer molecules which show activity cliff.

    """
    data_type = type_of_target(data[data_col].dropna().values, input_name=data_col)

    if data_type is None:
        raise ValueError(f"The column {data_col} contains less than 2 unique values.")

    mol_hash_ids_with_cliff = []
    if data_type in CONTINUOUS:
        # compute modified zscore
        data[f"{data_col}_zscore"] = modified_zscore(data[data_col].values)[0]

    for hashmol_id, group_df in data.groupby(groupby_col):
        if group_df.shape[0] > 1:
            if data_type in CONTINUOUS:
                has_cliff = (
                    group_df[f"{data_col}_zscore"].max() - group_df[f"{data_col}_zscore"].min()
                ) > threshold
            else:
                has_cliff = len(group_df[data_col].unique()) > 1

            if has_cliff:
                mol_hash_ids_with_cliff.append(hashmol_id)

    return mol_hash_ids_with_cliff


def _process_stereoisomer_with_activity_cliff(
    data: pd.DataFrame, data_cols: List[str], mask_stereo_undefined_mols: bool = True, **kwargs
):
    """Identify and discard the stereoisomers which the stereocenter are not well-defined but show activity cliff.

    Args:
        data: Parsed dataset
        data_cols: Data column names to be processed.
        mask_stereo_undefined_mols: Whether remove molecules and the bioactivity which shows chiral shift
                                    but stereo information not defined in the molecule representation.
        kwargs: Parameters for identify the activity cliff between the stereoisomers.
    """
    for data_col in data_cols:
        data_col_mask = (
            [data_col, data_col[len(CLASS_PREFIX) :]] if data_col.startswith(CLASS_PREFIX) else data_col
        )
        mol_with_cliff = _identify_stereoisomers_with_activity_cliff(data=data, data_col=data_col, **kwargs)
        data.loc[data[NO_STEREO_UNIQUE_ID].isin(mol_with_cliff), f"{data_col}_stereo_cliff"] = True
        if len(mol_with_cliff) > 0 and mask_stereo_undefined_mols:
            mol_ids = data.query(f"`{data_col}_stereo_cliff`==True & (`{UNDEF_ED}` | `{UNDEF_EZ}`) ")[
                UNIQUE_ID
            ].values
            data.loc[data[UNIQUE_ID].isin(mol_ids), data_col_mask] = None
    return data


def _merge_duplicates(
    data: pd.DataFrame, data_cols: List[str], merge_on=None, keep_all_rows: bool = False
) -> pd.DataFrame:
    """Merge molecules with multiple measurements.
       To be robust to the outliers, `median` is used to compute the average value cross all measurements.

    Args:
        data: Parsed dataset
        data_cols: Data column names to be processed
        merge_on: Column name to identify the duplicated molecules.
                  dy default, `molhash_id` is used to identify the replicated molecule.

    """
    if merge_on is None:
        merge_on = [UNIQUE_ID]

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
    prefix: str = CLASS_PREFIX,
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


def check_outliers(data, data_cols, method: str = "zscore", prefix="OUTLIER", **kwargs) -> pd.DataFrame:
    """Automatic detection of outliers of bioactivity

    Args:
        data: Parsed dataset
        data_cols: Data column names for outlier detections
        method: Algorithm name of outlier detection. The supported methods are
            "iso": IsolationForest
            "lof": LocalOutlierFactor
            "svm": OneClassSVM
            "ee": EllipticEnvelope
            "zscore": ZscoreOutlier
        kwargs: Additional parameters for the automatic outlier detection method.
                The parameters impact largely the detection therefore should be carefully chosen.
        prefix: Prefix for the boolean column which indicate whether the data point is an outlier.

    Returns:
        Dataframe with added boolean column for outliers.

    See also:
        <sklearn.ensemble.IsolationForest>
        <sklearn.svm.OneClassSVM>
        <sklearn.covariance.EllipticEnvelope>
        <sklearn.neighbors.LocalOutlierFactor>
        <polaris.curation.util.ZscoreOutlier>

    """
    for data_col in data_cols:
        outlier_col = f"{prefix}_{data_col}"
        data[f"{prefix}_{data_col}"] = None
        ind = data.index.values
        notna_ind = ind[data[data_col].notna()]
        data.loc[notna_ind, outlier_col] = False
        outlier_ind = outlier_detection(X=data[data_col].dropna().values, method=method, **kwargs)[:, 0]
        outlier_ind = notna_ind[outlier_ind]

        data.loc[outlier_ind, outlier_col] = True

        if len(outlier_ind) > 0:
            logger.warning(
                f"Detected {len(outlier_ind)} outliers for data column {data_col} using {method}. "
                f"Please revise the data and consider remove the outliers. "
            )
    return data


def run_data_curation(
    data: pd.DataFrame,
    data_cols: List[str],
    mask_stereo_undefined_mols: bool = False,
    ignore_stereo: bool = False,
    class_thresholds: Dict = None,
    outlier_params: Dict = None,
    activity_cliff_params: Dict = None,
) -> pd.DataFrame:
    """Perform curation on the measured values from biological assay

    Args:
        data: Dataframe which include molecule, measured values.
        data_cols: Column names for the data of interest.
        mask_stereo_undefined_mols: Whether mask the molecule stereo pairs which show activity cliff from dataset.
        ignore_stereo: Ignore the stereochemistry information from data curation.
        class_thresholds: Parameters to define class labels for the above listed `data_cols`.
        outlier_params: Parameters for outlier detection.
        activity_cliff_params: Parameter for identify the activity cliff between the stereoisomers.

    See Also:

    """
    data = check_outliers(data, data_cols, **outlier_params if outlier_params is not None else {})

    # User should deal with scaling and normalization on their end
    data = _merge_duplicates(
        data=data,
        data_cols=data_cols,
        merge_on=[NO_STEREO_UNIQUE_ID] if ignore_stereo else [UNIQUE_ID],
    )
    # class conversion
    if class_thresholds:
        data = _class_conversion(data, data_cols, class_thresholds, prefix=CLASS_PREFIX)
    if not ignore_stereo:
        # detect stereo activity cliff, keep or remove
        data = _process_stereoisomer_with_activity_cliff(
            data=data,
            data_cols=[
                f"{CLASS_PREFIX}{data_col}" if f"{CLASS_PREFIX}{data_col}" in data.columns else data_col
                for data_col in data_cols
            ]
            if class_thresholds
            else data_cols,
            mask_stereo_undefined_mols=mask_stereo_undefined_mols,
            **activity_cliff_params if activity_cliff_params is not None else {},
        )

    return data
