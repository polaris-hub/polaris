# Quickstart
Welcome to the Polaris Quickstart guide! This page will introduce you to core concepts and you'll submit a first result to a benchmark on the [Polaris Hub](https://www.polarishub.io).

## Installation
!!! warning "`polaris-lib` vs `polaris`"
    Be aware that the package name differs between _pip_ and _conda_.

Polaris can be installed via _pip_:

```bash
pip install polaris-lib
```

or _conda_: 
```bash
conda install -c conda-forge polaris
```

## Core concepts
Polaris explicitly distinguished **datasets** and **benchmarks**. 

- A _dataset_ is simply a tabular collection of data, storing datapoints in a row-wise manner. 
- A _benchmark_ defines the ML task and evaluation logic (e.g. split and metrics) for a dataset.

One dataset can therefore be associated with multiple benchmarks. 

## Login
To submit or upload artifacts to the [Polaris Hub](https://polarishub.io/) from the client, you must first authenticate yourself. If you don't have an account yet, you can create one [here](https://polarishub.io/sign-up).

You can do this via the following command in your terminal:

```bash
polaris login
```

or in Python: 
```py
from polaris.hub.client import PolarisHubClient

with PolarisHubClient() as client:
    client.login()
```

## Benchmark API
To get started, we will submit a result to the [`polaris/hello-world-benchmark`](https://polarishub.io/benchmarks/polaris/hello-world-benchmark).

```python
import polaris as po

# Load the benchmark from the Hub
benchmark = po.load_benchmark("polaris/hello-world-benchmark")

# Get the train and test data-loaders
train, test = benchmark.get_train_test_split()

# Use the training data to train your model
# Get the input as an array with 'train.inputs' and 'train.targets'  
# Or simply iterate over the train object.
for x, y in train:
    ...

# Work your magic to accurately predict the test set
predictions = [0.0 for x in test]

# Evaluate your predictions
results = benchmark.evaluate(predictions)

# Submit your results
results.upload_to_hub(owner="dummy-user")
```

Through immutable datasets and standardized benchmarks, Polaris aims to serve as a source of truth for machine learning in drug discovery. The limited flexibility might differ from your typical experience, but this is by design to improve reproducibility. Learn more [here](https://polarishub.io/blog/reproducible-machine-learning-in-drug-discovery-how-polaris-serves-as-a-single-source-of-truth).

## Dataset API
Loading a benchmark will automatically load the underlying dataset. We can also directly access the [`polaris/hello-world`](https://polarishub.io/datasets/polaris/hello-world) dataset.

```python
import polaris as po

# Load the dataset from the Hub
dataset = po.load_dataset("polaris/hello-world")

# Get information on the dataset size
dataset.size()

# Load a datapoint in memory
dataset.get_data(
    row=dataset.rows[0],
    col=dataset.columns[0],
)

# Or, similarly:
dataset[dataset.rows[0], dataset.columns[0]]

# Get an entire data point
dataset[0]
```

Drug discovery research involves a maze of file formats (e.g. PDB for 3D structures, SDF for small molecules, and so on). Each format requires specialized knowledge to parse and interpret properly. At Polaris, we wanted to remove that barrier. We use a universal data format based on [Zarr](https://zarr.dev/). Learn more [here](https://polarishub.io/blog/dataset-v2-built-to-scale).

## Where to next?

Now that you've seen how easy it is to use Polaris, let's dive into the details through [a set of tutorials](./tutorials/submit_to_benchmark.ipynb)!

---
