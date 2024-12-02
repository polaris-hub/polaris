from collections import defaultdict
from typing import TYPE_CHECKING
import os

import pandas as pd
import zarr

from polaris.dataset import ColumnAnnotation
from polaris.dataset.converters._base import Converter, FactoryProduct

if TYPE_CHECKING:
    from polaris.dataset import DatasetFactory


class ZarrConverter(Converter):
    """Parse a [.zarr](https://zarr.readthedocs.io/en/stable/index.html) archive into a Polaris `Dataset`.

    Tip: Tutorial
        To learn more about the zarr format, see the
        [tutorial](../tutorials/dataset_zarr.ipynb).

    Warning: Loading from `.zarr`
        Loading and saving datasets from and to `.zarr` is still experimental and currently not
        fully supported by the Hub.

    A `.zarr` file can contain groups and arrays, where each group can again contain groups and arrays.
    Within Polaris, the Zarr archive is expected to have a flat hierarchy where each array corresponds
    to a single column and each array contains the values for all datapoints in that column.
    """

    def convert(self, path: str, factory: "DatasetFactory", append: bool = False) -> FactoryProduct:
        src = zarr.open(path, "r")

        v = next(src.group_keys(), None)
        if v is not None:
            raise ValueError("The root of the zarr hierarchy should only contain arrays.")

        # Copy to the source zarr, so everything is in one place
        pointer_start_dict = {col: 0 for col, _ in src.arrays()}
        if append:
            if not os.path.exists(factory.zarr_root.store.path):
                raise RuntimeError(
                    f"Zarr store {factory.zarr_root.store.path} doesn't exist. \
                    Please make sure the zarr store {factory.zarr_root.store.path} is created. Or set `append` to `False`."
                )
            else:
                for col, arr in src.arrays():
                    pointer_start_dict[col] += factory.zarr_root[col].shape[0]
                    factory.zarr_root[col].append(arr)
        else:
            zarr.copy_store(source=src.store, dest=factory.zarr_root.store, if_exists="skip")

        # Construct the table
        # Parse any group into a column
        data = defaultdict(dict)
        for col, arr in src.arrays():
            for i in range(len(arr)):
                data[col][i] = self.get_pointer(arr.name.removeprefix("/"), i)

        # Construct the dataset
        table = pd.DataFrame(data)
        return table, {k: ColumnAnnotation(is_pointer=True) for k in table.columns}, {}
