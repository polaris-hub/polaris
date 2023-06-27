# Polaris

[![test](https://github.com/datamol-io/polaris/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/polaris/actions/workflows/test.yml)
[![release](https://github.com/datamol-io/polaris/actions/workflows/release.yml/badge.svg)](https://github.com/datamol-io/polaris/actions/workflows/release.yml)
[![code-check](https://github.com/datamol-io/polaris/actions/workflows/code-check.yml/badge.svg)](https://github.com/datamol-io/polaris/actions/workflows/code-check.yml)
[![doc](https://github.com/datamol-io/polaris/actions/workflows/doc.yml/badge.svg)](https://github.com/datamol-io/polaris/actions/workflows/doc.yml)

_So many stars in the sky, yet one is enough to guide you home._

Foster the development of impactful AI models in drug discovery.

```python
import polaris as po

benchmark = po.load_benchmark("/path/to/benchmark")
train, test = benchmark.get_train_test_split()

# Work your magic

benchmark.evaluate(...)
```

## Development lifecycle

### Setup dev environment

```bash
micromamba create -n polaris -f env.yml
micromamba activate polaris

pip install -e .
```

### Tests

You can run tests locally with:

```bash
pytest
```

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).
