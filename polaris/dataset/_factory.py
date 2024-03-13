import os
from typing import Dict, Optional

import datamol as dm
import pandas as pd
import zarr

from polaris.dataset import ColumnAnnotation, Dataset
from polaris.dataset.converters import SDFConverter, ZarrConverter


def get_dataset_from_file(path: str, zarr_root_path: Optional[str] = None) -> Dataset:
    """
    This function is a convenience function to create a dataset from a file.
    It uses the factory design pattern to create the dataset.
    For more complicated datasets, please use the `DatasetFactory` directly.
    """
    factory = DatasetFactory(zarr_root_path=zarr_root_path)
    factory.register_converter("sdf", SDFConverter())
    factory.register_converter("zarr", ZarrConverter())

    factory.add_from_file(path)
    return factory.build()


class DatasetFactory:
    """
    The DatasetFactory is meant to more easily create complex datasets.
    It uses the factory design pattern.
    """

    def __init__(self, zarr_root_path: Optional[str] = None) -> None:
        self.zarr_root_path = os.path.abspath(zarr_root_path).rstrip("/")
        self._zarr_root = None

        self.table: pd.DataFrame = pd.DataFrame()
        self.annotations: Dict[str, ColumnAnnotation] = {}

        self._converters = {}

    @property
    def zarr_root(self) -> zarr.Group:
        if self.zarr_root_path is None:
            raise ValueError("You need to pass `zarr_root_path` to the factory to use pointer columns")

        if self._zarr_root is None:
            self._zarr_root = zarr.open(self.zarr_root_path, "w")
            if not isinstance(self._zarr_root, zarr.Group):
                raise ValueError("The root of the zarr hierarchy should be a group")
        return self._zarr_root

    def register_converter(self, ext: str, converter):
        self._converters[ext] = converter

    def reset(self):
        self.table = pd.DataFrame()
        self.annotations = {}

    def add_column(
        self,
        column: pd.Series,
        annotation: Optional[ColumnAnnotation] = None,
    ):
        """Adds a single column"""
        if column.name is None:
            raise RuntimeError("You need to specify a column name")
        if column.name in self.table.columns:
            raise ValueError(f"Column name '{column.name}' already exists in the table")

        if annotation is not None and annotation.is_pointer:
            if self.zarr_root is None:
                raise ValueError("You need to pass `zarr_root_path` to the factory to use pointer columns")

        self.table[column.name] = column

        if annotation is None:
            annotation = ColumnAnnotation()
        self.annotations[column.name] = annotation

    def add_columns(
        self,
        df: pd.DataFrame,
        annotations: Optional[Dict[str, ColumnAnnotation]] = None,
        merge_on: Optional[str] = None,
    ):
        """Adds a single column"""
        if merge_on is not None:
            df = self.table.merge(df, on=merge_on, how="outer")

        if annotations is None:
            annotations = {}
        annotations = {**self.annotations, **annotations}

        if merge_on is not None:
            self.reset()

        for name, series in df.items():
            annotation = annotations.get(name)
            self.add_column(series, annotation)

    def add_from_file(self, path: str):
        ext = dm.fs.get_extension(path)
        converter = self._converters.get(ext)
        if converter is None:
            raise ValueError(f"No converter found for extension {ext}")

        table, annotations = converter.convert(path, self)
        self.add_columns(table, annotations)

    def build(self) -> Dataset:
        return Dataset(table=self.table, annotations=self.annotations)
