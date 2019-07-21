# riskparity.py

fast and scalable design of risk parity portfolios in Python

[![PyPI version](https://badge.fury.io/py/riskparityportfolio.svg)](https://badge.fury.io/py/riskparityportfolio)
[![Downloads](https://pepy.tech/badge/riskparityportfolio)](https://pepy.tech/project/riskparityportfolio)

[![Travis (.org)](https://img.shields.io/travis/mirca/riskparity.py.svg?label=travis-ci&style=flat-square)](https://travis-ci.org/mirca/riskparity.py)
[![codecov](https://codecov.io/gh/mirca/riskparity.py/branch/master/graph/badge.svg)](https://codecov.io/gh/mirca/riskparity.py)


**riskparityportfolio** provides tools to design risk parity portfolios.
In its simplest form, we consider the convex formulation with a unique solution proposed by
[Spinu (2013)](https://dx.doi.org/10.2139/ssrn.2297383) and use a cyclical method inspired by
[Griveau-Billion (2013)](https://arxiv.org/pdf/1311.4057.pdf). For more general formulations,
which are usually nonconvex, we implement the successive convex approximation
method proposed by [Feng & Palomar (2015)](https://doi.org/10.1109/TSP.2015.2452219).

For the R version of this library,
check out: [https://mirca.github.io/riskParityPortfolio](https://mirca.github.io/riskParityPortfolio).

## Installation

### Dependencies
`riskparityportfolio` depends on `numpy`, `tensorflow2`, `quadprog`,
`pybind`, and `tqdm`, which can be installed via `pip`.

The *stable* version of `riskparityportfolio` can be installed via `pip`:
```
$ pip install riskparityportfolio
```

The *development* version of `riskparityportfolio` can be installed as:
```
$ git clone https://github.com/mirca/riskparity.py
$ cd riskparity.py
$ pip install -e .
```

## Basic usage
```{python}
import numpy as np
import riskparityportfolio as rpp
np.random.seed(42)

# creates a correlation matrix from time-series of five assets
x = np.random.normal(size=1000).reshape((5, -1))
S = x @ x.T

# create the desired risk budgeting vector
b = np.ones(len(S)) / len(S)

# design the portfolio
w = rpp.vanilla.design(S, b)
print(w)

# compute the risk budgeting
rc = w @ (corr * w)
print(rc / np.sum(rc))

# let's try a different budget
b = np.array([0.01, 0.09, .1, .1, .7])
w = rpp.design(corr, b)
print(w)
rc = w @ (corr * w)
print(rc / np.sum(rc))
```
