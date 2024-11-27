<div align="center">
    <img src="docs/images/logo.svg" width="300px">
</div>
</br>

<p align="center">
    <a href="https://polarishub.io/" target="_blank">
      ✨ Polaris Hub
  </a> |
  <a href="https://polaris-hub.github.io/polaris/" target="_blank">
      📚 Client Doc
  </a>
</p>

---

|  |  | 
| --- | --- | 
| Latest Release | [![PyPI](https://img.shields.io/pypi/v/polaris-lib)](https://pypi.org/project/polaris-lib/) | 
|  | [![Conda](https://img.shields.io/conda/v/conda-forge/polaris?label=conda&color=success)](https://anaconda.org/conda-forge/polaris) |
| Python Version | [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/polaris-lib)](https://pypi.org/project/polaris-lib/) |
| License | [![Code license](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/polaris-hub/polaris/blob/main/LICENSE) |
| Downloads | [![PyPI - Downloads](https://img.shields.io/pypi/dm/polaris-lib)](https://pypi.org/project/polaris-lib/) |
| | [![Conda](https://img.shields.io/conda/dn/conda-forge/polaris)](https://anaconda.org/conda-forge/polaris) |
| Citation | [![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42256--024--00911--w-blue)](https://doi.org/10.1038/s42256-024-00911-w) |

Polaris establishes a novel, industry‑certified standard to foster the development of impactful methods in AI-based drug discovery.

This library is a Python client to interact with the [Polaris Hub](https://polarishub.io/). It allows you to:

- Download Polaris datasets and benchmarks.
- Evaluate a custom method against a Polaris benchmark.
- Create and upload new datasets and benchmarks.

## Quick API Tour

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

## Documentation

Please refer to the [documentation](https://polaris-hub.github.io/polaris/), which contains tutorials for getting started with `polaris` and detailed descriptions of the functions provided.

## How to cite

Please cite Polaris if you use it in your research. A list of relevant publications: 

- [![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2024--6dbwv--v2-blue)](https://doi.org/10.26434/chemrxiv-2024-6dbwv-v2) - Preprint, Method Comparison Guidelines.
- [![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42256--024--00911--w-blue)](https://doi.org/10.1038/s42256-024-00911-w) - Nature Correspondence, Call to Action.
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13652587.svg)](https://doi.org/10.5281/zenodo.13652587) - Zenodo, Code Repository.

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

```shell
conda env create -n polaris -f env.yml
conda activate polaris

pip install --no-deps -e .
```

<details>
  <summary>Other installation options</summary>
  
    Alternatively, using [uv](https://github.com/astral-sh/uv):
    ```shell
    uv venv -p 3.12 polaris
    source .venv/polaris/bin/activate
    uv pip compile pyproject.toml -o requirements.txt --all-extras
    uv pip install -r requirements.txt 
    ```   
</details>


### Tests

You can run tests locally with:

```shell
pytest
```

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).
