from collections import defaultdict
from typing import TYPE_CHECKING

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

    def convert(self, path: str, factory: "DatasetFactory") -> FactoryProduct:
        src = zarr.open(path, "r")

        v = next(src.group_keys(), None)
        if v is not None:
            raise ValueError("The root of the zarr hierarchy should only contain arrays.")

        # Copy to the source zarr, so everything is in one place
        zarr.copy_all(source=src, dest=factory.zarr_root)

        # Construct the table
        # Parse any group into a column
        data = defaultdict(dict)
        for col, arr in src.arrays():
            for i in range(len(arr)):
                data[col][i] = self.get_pointer(arr.name.removeprefix("/"), i)

        # Construct the dataset
        table = pd.DataFrame(data)
        return table, {k: ColumnAnnotation(is_pointer=True) for k in table.columns}, {}
