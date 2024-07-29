import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from biopandas.pdb import PandasPdb

def create_zarr_from_pdb(pdb_file, zarr_file, mode="w", index=0):

    # load structure
    ppdb_df = PandasPdb().read_pdb(pdb_file)

    # Create a Zarr store
    store = zarr.DirectoryStore(zarr_file)

    if mode =='w':
        root = zarr.open_group(store=store, mode='a') 
    else:
        # Create a root group
        root = zarr.group(store=store)

    protein = root.create_group(index)

    dtype_dict = {'object': 'str'}

    for key in ['ATOM', 'HETATM', 'ANISOU', 'OTHERS']:

        # Create group and add datasets
        group = protein.create_group(key)
        
        # create a dataset  
        group.create_dataset("column_names", data=ppdb_df.df[key].columns.tolist(), dtype="str")

        for col_name, col_val in ppdb_df.df[key].items():
            dtype = col_val.values.dtype
            dtype = dtype_dict.get(str(dtype) ,dtype)
            group.create_dataset(col_name, data=col_val.values, dtype=dtype)
