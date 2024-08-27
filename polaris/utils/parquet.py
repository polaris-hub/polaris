import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_from_dataframe(table, path_to_file, file_name_suffix):
    """Function that leverage Pyarrow to write a Pandas dataframe
    to disk as a Parquet file"""

    # Write dataframe to parquet file
    PARQUET_PATH = f"{path_to_file}/file_{file_name_suffix}.parquet"
    pq.write_table(table, PARQUET_PATH)

    # Return path to file
    return PARQUET_PATH


def append_parquet_files(input_files, output_file):
    """Combine numerous Parquet files (of the same schema) into a single
    Parquet file. This function attempts to be memory efficient via
    sequentially reading one Parquet file into memory and appending it to
    an output Parquet on disk."""

    # Create a writer object for the output Parquet file
    writer = None

    # Loop through the input files & append them to the final parquet
    for file in input_files:
        #
        # Read the current Parquet file into a Table
        table = pq.read_table(file)

        if writer is None:
            #
            # Create the writer with the schema from the first file
            writer = pq.ParquetWriter(output_file, table.schema)

        # Write the table to the output file
        writer.write_table(table)

    # Close the writer to finalize the file
    if writer:
        writer.close()
