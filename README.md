<h1 align="center">Polaris</h1>
<p align="center"><i>So many stars in the sky, yet one is enough to guide you home.</i></p>

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

[![PyPI](https://img.shields.io/pypi/v/polaris-lib)](https://pypi.org/project/polaris-lib/)
[![Conda](https://img.shields.io/conda/v/conda-forge/polaris?label=conda&color=success)](https://anaconda.org/conda-forge/polaris)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/polaris-lib)](https://pypi.org/project/polaris-lib/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/polaris)](https://anaconda.org/conda-forge/polaris)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/polaris-lib)](https://pypi.org/project/polaris-lib/)
[![Code license](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/polaris-hub/polaris/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/polaris-hub/polaris)](https://github.com/polaris-hub/polaris/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/polaris-hub/polaris)](https://github.com/polaris-hub/polaris/network/members)

[![test](https://github.com/polaris-hub/polaris/actions/workflows/test.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/test.yml)
[![release](https://github.com/polaris-hub/polaris/actions/workflows/release.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/release.yml)
[![code-check](https://github.com/polaris-hub/polaris/actions/workflows/code-check.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/code-check.yml)
[![doc](https://github.com/polaris-hub/polaris/actions/workflows/doc.yml/badge.svg)](https://github.com/polaris-hub/polaris/actions/workflows/doc.yml)

Polaris establishes a novel, industry‑certified standard to foster the development of impactful methods in AI-based drug discovery.

This library is a Python client to interact with the [Polaris Hub](https://polarishub.io/). It allows you to:

- Download Polaris datasets and benchmarks.
- Evaluate a custom method against a Polaris benchmark.
- Create and upload new datasets and benchmarks.

> [!WARNING]
> The Polaris Hub is currently released as a closed, private beta. We hope to officially release it early 2024.


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

## Installation

You can install `polaris` using conda/mamba/micromamba:

```bash
conda install -c conda-forge polaris
```

You can also use pip:

```bash
pip install polaris-lib
```

## Development lifecycle

### Setup dev environment

```bash
conda env create -n polaris -f env.yml
conda activate polaris

pip install --no-deps -e .
```

### Tests

You can run tests locally with:

```bash
pytest
```

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).
