import os
from typing import Literal

import datamol as dm
import pandas as pd
import zarr
from loguru import logger

from polaris.dataset import ColumnAnnotation, DatasetV1
from polaris.dataset._adapters import Adapter
from polaris.dataset.converters import Converter, PDBConverter, SDFConverter, ZarrConverter


def create_dataset_from_file(path: str, zarr_root_path: str | None = None) -> DatasetV1:
    """
    This function is a convenience function to create a dataset from a file.

    It sets up the dataset factory with sensible defaults for the converters.
    For creating more complicated datasets, please use the `DatasetFactory` directly.
    """
    factory = DatasetFactory(zarr_root_path=zarr_root_path)
    factory.register_converter("sdf", SDFConverter())
    factory.register_converter("zarr", ZarrConverter())
    factory.register_converter("pdb", PDBConverter())

    factory.add_from_file(path)
    return factory.build()


def create_dataset_from_files(
    paths: list[str], zarr_root_path: str | None = None, axis: Literal[0, 1, "index", "columns"] = 0
) -> DatasetV1:
    """
    This function is a convenience function to create a dataset from multiple files.

    It sets up the dataset factory with sensible defaults for the converters.
    For creating more complicated datasets, please use the `DatasetFactory` directly.

    Args:
        axis: Axis along which the files should be added.
            - 0 or 'index': append the rows with files. Files must be of the same type.
            - 1 or 'columns': append the columns with files. Files can be of the different types.
    """
    factory = DatasetFactory(zarr_root_path=zarr_root_path)
    factory.register_converter("sdf", SDFConverter())
    factory.register_converter("zarr", ZarrConverter())
    factory.register_converter("pdb", PDBConverter())

    factory.add_from_files(paths, axis)
    return factory.build()


class DatasetFactory:
    """
    The `DatasetFactory` makes it easier to create complex datasets.

    It is based on the factory design pattern and allows a user to specify specific handlers
    (i.e. [`Converter`][polaris.dataset.converters._base.Converter] objects) for different file types.
    These converters are used to convert commonly used file types in drug discovery
    to something that can be used within Polaris while losing as little information as possible.

    In addition, it contains utility method to incrementally build out a dataset from different sources.

    Tip: Try quickly converting one of your datasets
        The `DatasetFactory` is designed to give you full control.
        If your dataset is saved in a single file and you don't need anything fancy, you can try use
        [`create_dataset_from_file`][polaris.dataset.create_dataset_from_file] instead.

        ```py
        from polaris.dataset import create_dataset_from_file
        dataset = create_dataset_from_file("path/to/my_dataset.sdf")
        ```

    Question: How to make adding meta-data easier?
        The `DatasetFactory` is designed to more easily pull together data from different sources.
        However, adding meta-data remains a laborious process. How could we make this simpler through
        the Python API?
    """

    def __init__(
        self, zarr_root_path: str | None = None, converters: dict[str, Converter] | None = None
    ) -> None:
        """
        Create a new factory object.

        Args:
            zarr_root_path: The root path of the zarr hierarchy. If you want to use pointer columns,
                this arguments needs to be passed.
            converters: The converters to use for specific file types.
                You can also register them later with register_converter().
        """

        if converters is None:
            converters = {}

        self._converters: dict[str, Converter] = converters
        self.reset(zarr_root_path=zarr_root_path)

    @property
    def zarr_root_path(self) -> zarr.Group:
        """
        The root of the zarr archive for the Dataset that is being built.
        All data for a single dataset is expected to be stored in the same Zarr archive.
        """
        if self._zarr_root_path is None:
            raise ValueError("You need to pass `zarr_root_path` to the factory to use pointer columns")
        return self._zarr_root_path

    @property
    def zarr_root(self) -> zarr.Group:
        """
        The root of the zarr archive for the Dataset that is being built.
        All data for a single dataset is expected to be stored in the same Zarr archive.
        """
        if self._zarr_root is None:
            # NOTE (cwognum): The DirectoryStore is the default store when calling zarr.open
            #   I nevertheless explicitly set it here to make it clear that this is a design decision.
            #   We could consider using different stores, such as the NestedDirectoryStore.
            store = zarr.DirectoryStore(self.zarr_root_path)
            self._zarr_root = zarr.open(store, "w")
            if not isinstance(self._zarr_root, zarr.Group):
                raise ValueError("The root of the zarr hierarchy should be a group")
        return self._zarr_root

    def register_converter(self, ext: str, converter: Converter):
        """
        Registers a new converter for a specific file type.

        Args:
            ext: The file extension for which the converter should be used.
                There can only be a single converter per file extension.
            converter: The handler for the file type. This should convert
                the file to a Polaris-compatible format.
        """
        if ext in self._converters:
            logger.info(f"You are overwriting the converter for the {ext} extension.")
        self._converters[ext] = converter

    def add_column(
        self,
        column: pd.Series,
        annotation: ColumnAnnotation | None = None,
        adapters: Adapter | None = None,
    ):
        """
        Add a single column to the DataFrame

        We require:

        1. The name attribute of the column to be set.
        2. The name attribute of the column to be unique.
        3. If the column is a pointer column, the `zarr_root_path` needs to be set.
        4. The length of the column to match the length of the already constructed table.

        Args:
            column: The column to add to the dataset.
            annotation: The annotation for the column. If None, a default annotation will be used.
        """

        # Verify the column can be added
        if column.name is None:
            raise ValueError("You need to specify a column name")
        if column.name in self._table.columns:
            raise ValueError(f"Column name '{column.name}' already exists in the table")
        if not self._table.empty and len(column) != len(self._table):
            raise ValueError("The length of the column does not match the length of the table")

        if annotation is not None and annotation.is_pointer:
            if self.zarr_root is None:
                raise ValueError("You need to pass `zarr_root_path` to the factory to use pointer columns")

        # Actually add the column
        self._table[column.name] = column

        if annotation is None:
            annotation = ColumnAnnotation()
        self._annotations[column.name] = annotation

        if adapters is not None:
            self._adapters[column.name] = adapters

    def add_columns(
        self,
        df: pd.DataFrame,
        annotations: dict[str, ColumnAnnotation] | None = None,
        adapters: dict[str, Adapter] | None = None,
        merge_on: str | None = None,
    ):
        """
        Add multiple columns to the dataset based on another dataframe.

        To have more control over how the two dataframes are combined, you can
        specify a column to merge on. This will always do an **outer** join.

        If not specifying a key to merge on, the columns will simply be added to the dataset
        that has been built so far without any reordering. They are therefore expected to meet all
        the same expectations as for [`add_column`][polaris.dataset.DatasetFactory.add_column].

        Args:
            df: A Pandas DataFrame with the columns that we want to add to the dataset.
            annotations: The annotations for the columns. If None, default annotations will be used.
            merge_on: The column to merge on, if any.
        """

        if merge_on is not None:
            df = self._table.merge(df, on=merge_on, how="outer")

        if annotations is None:
            annotations = {}
        annotations = {**self._annotations, **annotations}

        if adapters is None:
            adapters = {}
        adapters = {**self._adapters, **adapters}

        if merge_on is not None:
            self.reset(self._zarr_root_path)

        for name, series in df.items():
            annotation = annotations.get(name)
            adapter = adapters.get(name)
            self.add_column(series, annotation, adapter)

    def add_from_file(self, path: str):
        """
        Uses the registered converters to parse the data from a specific file and add it to the dataset.
        If no converter is found for the file extension, it raises an error.

        Args:
            path: The path to the file that should be parsed.
        """
        ext = dm.fs.get_extension(path)
        converter = self._converters.get(ext)
        if converter is None:
            raise ValueError(f"No converter found for extension {ext}")

        table, annotations, adapters = converter.convert(path, self)
        self.add_columns(table, annotations, adapters)

    def add_from_files(self, paths: list[str], axis: Literal[0, 1, "index", "columns"]):
        """
        Uses the registered converters to parse the data from a specific files and add them to the dataset.
        If no converter is found for the file extension, it raises an error.

        Args:
            paths: The list of paths that should be parsed.
            axis: Axis along which the files should be added.
                - 0 or 'index': append the rows with files. Files must be of the same type.
                - 1 or 'columns': append the columns with files. Files can be of the different types.
        """
        if axis in [0, "index"]:
            ext = dm.fs.get_extension(paths[0])
            converter = self._converters.get(ext)
            if converter is None:
                raise ValueError(f"No converter found for extension {ext}")

            tables = []
            append = False
            for path in paths:
                table, annotations, adapters = converter.convert(path, self, append=append)
                tables.append(table)
                append = True
            self.add_columns(pd.concat(tables, axis=0, ignore_index=True), annotations, adapters)
        else:
            for path in paths:
                self.add_from_file(path)

    def build(self) -> DatasetV1:
        """Returns a Dataset based on the current state of the factory."""
        zarr.consolidate_metadata(self.zarr_root.store)
        return DatasetV1(
            table=self._table,
            annotations=self._annotations,
            default_adapters=self._adapters,
            zarr_root_path=self.zarr_root_path,
        )

    def reset(self, zarr_root_path: str | None = None):
        """
        Resets the factory to its initial state to start building the next dataset from scratch.
        Note that this will not reset the registered converters.

        Args:
            zarr_root_path: The root path of the zarr hierarchy. If you want to use pointer columns
                for your next dataset, this arguments needs to be passed.
        """

        if zarr_root_path is not None:
            zarr_root_path = os.path.abspath(zarr_root_path).rstrip("/")

        self._zarr_root = None
        self._zarr_root_path = zarr_root_path
        self._table = pd.DataFrame()
        self._annotations = {}
        self._adapters = {}
