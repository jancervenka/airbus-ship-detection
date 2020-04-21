# Airbus Ship Detection Challenge

[1]: https://www.kaggle.com/c/airbus-ship-detection/

This repository contains my research of the [Airbus Ship Detection Challenge][1].

Version: `0.1.0`

## Installation

The package can be installed using the `setup.py`.

```bash
python setup.py install
```

## Building the Image

The package can be build as a Docker image.

```
sudo docker build . -t asdc
```

## Tests

Unit tests can run using `pytest`. `FutureWarning` and `DeprecationWarning` warnings
are silenced.

```bash
python -m pytest asdc/tests
```