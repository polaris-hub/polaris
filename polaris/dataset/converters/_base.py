import abc
from typing import Dict, Tuple, TypeAlias, Union

import pandas as pd

from polaris.dataset import ColumnAnnotation
from polaris.dataset._adapters import Adapter
from polaris.dataset._dataset import _INDEX_SEP

FactoryProduct: TypeAlias = Tuple[pd.DataFrame, Dict[str, ColumnAnnotation], Dict[str, Adapter]]


class Converter(abc.ABC):
    @abc.abstractmethod
    def convert(self, path: str) -> FactoryProduct:
        """This converts a file into a table and possibly annotations"""
        raise NotImplementedError

    @staticmethod
    def get_pointer(column: str, index: Union[int, slice]) -> str:
        """
        Creates a pointer.

        Args:
            column: The name of the column. Each column has its own group in the root.
            index: The index or slice of the pointer.
        """
        if isinstance(index, slice):
            index_substr = f"{_INDEX_SEP}{index.start}:{index.stop}"
        else:
            index_substr = f"{_INDEX_SEP}{index}"
        return f"{column}{index_substr}"
