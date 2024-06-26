# Quickstart

## Installation

First things first, let's install Polaris!

We highly recommend using a [Conda Python distribution](https://github.com/conda-forge/miniforge), such as `mamba`:

```bash
mamba install -c conda-forge polaris
```

??? info "Other installation options"
You can replace `mamba` by `conda`. The package is also pip installable if you need it: `pip install polaris-lib`.

## Benchmarking API

At its core, Polaris is a benchmarking library. It provides a simple API to run benchmarks. While it can be used
independently, it is built to easily integrate with the [Polaris Hub](https://polarishub.io/). The hub hosts
a variety of high-quality datasets, benchmarks and associated results.

If all you care about is to partake in a benchmark that is hosted on the hub, it is as simple as:

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

That's all there is to it to partake in a benchmark. No complicated, custom data-loaders or evaluation protocol. With just a few lines of code, you can feel confident that you are properly evaluating your model and focus on what you do best: Solving the hard problems in our domain!

Similarly, you can easily access a dataset.

```python
import polaris as po

# Load the dataset from the hub
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

# Get the first 10 rows in memory
dataset[:10]
```

## Core concepts

At the core of our API are 4 core concepts, each associated with a class:

1. [`Dataset`][polaris.dataset.Dataset]: The dataset class is carefully designed data-structure, stress-tested on terra-bytes of data, to ensure whatever dataset you can think of, you can easily create, store and use it.
2. [`BenchmarkSpecification`][polaris.benchmark.BenchmarkSpecification]: The benchmark specification class wraps a `Dataset` with additional meta-data to produce a the benchmark. Specifically, it specifies how to evaluate a model's performance on the underlying dataset (e.g. the train-test split and metrics). It provides a simple API to run said evaluation protocol.
3. [`Subset`][polaris.dataset.Subset]: The subset class should be used as a starting-point for any framework-specific (e.g. PyTorch or Tensorflow) data loaders. To facilitate this, it abstracts away the non-trivial logic of accessing the data and provides several style of access to built upon.
4. [`BenchmarkResults`][polaris.evaluate.BenchmarkResults]: The benchmark results class stores the results of a benchmark, along with additional meta-data. This object can be easily uploaded to the Polaris Hub and shared with the broader community.

## Where to next?

Now that you've seen how easy it is to use Polaris, let's dive into the details through a set of tutorials!

---
