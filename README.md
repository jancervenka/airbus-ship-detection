# Airbus Ship Detection Challenge

[1]: https://www.kaggle.com/c/airbus-ship-detection/

This repository contains my research of the [Airbus Ship Detection Challenge][1].

Version: `0.1.0`

## Project

The project consists of two main packages `core` and `service`.

The `core` package defines the model architecture, optimization and training process.

The `service` package defines and implements RESTful API and backend
processor along with a data communication layer connecting the two components.

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

Unit tests can be run using `pytest`. `FutureWarning` and `DeprecationWarning` warnings
are silenced.

```bash
python -m pytest asdc/tests
```