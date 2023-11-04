<h1 align="center">Polaris</h1>
<h4 align="center">Polaris establishes a novel, industry‑certified standard to foster the development of impactful methods in AI-based drug discovery.</h4>

</br>
<div align="center">
    <img src="docs/images/logo-black.svg" width="100px">
</div>
</br>

<p align="center">
    <a href="https://polarishub.io/" target="_blank">
      🟆 Polaris Hub
  </a> |
  <a href="https://polaris-hub.github.io/polaris/" target="_blank">
      🛈 Client Doc
  </a>
</p>

---

[![PyPI](https://img.shields.io/pypi/v/splito)](https://pypi.org/project/splito/)
[![Conda](https://img.shields.io/conda/v/conda-forge/splito?label=conda&color=success)](https://anaconda.org/conda-forge/splito)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/splito)](https://pypi.org/project/splito/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/splito)](https://anaconda.org/conda-forge/splito)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/splito)](https://pypi.org/project/splito/)
[![Code license](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/datamol-io/splito/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/datamol-io/splito)](https://github.com/datamol-io/splito/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/datamol-io/splito)](https://github.com/datamol-io/splito/network/members)

[![test](https://github.com/polaris-hub/polaris/actions/workflows/test.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/test.yml)
[![release](https://github.com/polaris-hub/polaris/actions/workflows/release.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/release.yml)
[![code-check](https://github.com/polaris-hub/polaris/actions/workflows/code-check.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/code-check.yml)
[![doc](https://github.com/polaris-hub/polaris/actions/workflows/doc.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/doc.yml)

_So many stars in the sky, yet one is enough to guide you home._

This library is a Python client to interact with the [Polaris Hub](https://polarishub.io/). It allows to:

- Download Polaris datasets and benchmarks.
- Evaluate a custom method against a Polaris benchmark.
- Create and upload new datasets and benchmarks.

## Quick API Tour

```python
import polaris as po

# Download a benchmark (the associated dataset will be transparently downloaded)
benchmark = po.load_benchmark("org_or_user/name")

# Retrieve the splits
train, test = benchmark.get_train_test_split()

# Work your magic!
y_pred = ...

# Run the evaluation procedure
results = benchmark.evaluate(y_pred)

# Upload your results to the hub
results.upload_to_hub()
```

## Documentation

Please refer to the [documentation](https://polaris-hub.github.io/polaris/), which contains tutorials for getting started with `polaris` and detailed descriptions of the functions provided.

## Installation - Private MVP

_**Important:** Since Polaris is not yet public, you must clone this repository before installing the `polaris` client._

With conda/mamba/micromamba:

```bash
# Clone the repo
git clone https://github.com/polaris-hub/polaris.git

# Install the deps in a new conda env
micromamba create -n polaris -f polaris/env.yml

# Install the polaris client
micromamba run -n polaris pip install --no-deps polaris/

# Test it works
micromamba run -n polaris polaris --help
```

With pip:

```bash
# Clone the repo
git clone https://github.com/polaris-hub/polaris.git

# Install the polaris client
pip install polaris/

# Test it works
polaris --help
```

## Installation - Note Yet Working

You can install `polaris` using conda/mamba/micromamba:

```bash
mamba install -c conda-forge polaris
```

You can also use pip:

```bash
pip install polaris-py
```

## Development lifecycle

### Setup dev environment

```bash
micromamba create -n polaris -f env.yml
micromamba activate polaris

pip install --no-deps -e .
```

### Tests

You can run tests locally with:

```bash
pytest
```

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).
