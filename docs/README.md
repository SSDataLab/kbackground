# KBackground
<a href="https://github.com/ssdatalab/kbackground/actions/workflows/tests.yml"><img src="https://github.com/ssdatalab/kbackground/workflows/pytest/badge.svg" alt="Test status"/></a> <a href="https://github.com/ssdatalab/kbackground/actions/workflows/flake8.yml"><img src="https://github.com/ssdatalab/kbackground/workflows/flake8/badge.svg" alt="flake8 status"/></a>[![Generic badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://ssdatalab.github.io/kbackground)

KBackground is a standalone Python tool to quickly model the rolling band in Kepler and K2 data.

## Installation

```
pip install kbackground
```

## Usage

To use `kbackground`, you can create an object using row, column and flux values. That object can then be used to provide the modeled background.

```python
from kbackground import Estimator
e = Estimator(time, row, column, flux)
bkg_model = e.model
```

`bkg_model` will be a 2D array with the same shape as `flux`.
