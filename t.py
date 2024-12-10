import polaris as po

# Load the dataset from the Hub
dataset = po.load_dataset("polaris/posebusters-v1")

# Get information on the dataset size
dataset.size()

# Load a datapoint in memory
dataset.get_data(
    row=dataset.rows[0],
    col=dataset.columns[0],
)

# Or, similarly:
dataset[dataset.rows[0], dataset.columns[0]]

# Get an entire row
dataset[0]
