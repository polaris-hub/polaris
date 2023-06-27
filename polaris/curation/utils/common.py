from typing import List
import pandas as pd


def copy_columns(data: pd.DataFrame, data_cols: List[str], prefix="ori") -> pd.DataFrame:
    """Copy and backup the original data columns before curation steps
    Args:
        data: Parsed dataset
        data_cols: Data column names to be copied
        prefix: Prefix for the back upped columns
    """
    for data_col in data_cols:
        data[f"{prefix}_{data_col}"] = data[data_col].copy()
    return data
