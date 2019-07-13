# riskparity.py

fast and scalable design of risk parity portfolios in Python

[![PyPI version](https://badge.fury.io/py/riskparityportfolio.svg)](https://badge.fury.io/py/riskparityportfolio)
[![Downloads](https://pepy.tech/badge/riskparityportfolio)](https://pepy.tech/project/riskparityportfolio)

## Installation
Via `pip`:
```
$ pip install riskparityportfolio
```

Via source:
```
$ git clone https://github.com/mirca/riskparity.py
$ cd python
$ pip install -e .
```

## Basic usage
```{python}
import numpy as np
import riskparityportfolio as rpp
np.random.seed(42)

# creates a correlation matrix from time-series of five assets
x = np.random.normal(size=1000).reshape((5, -1))
corr = x @ x.T

# create the desired risk budgeting vector
b = np.ones(len(corr)) / len(corr)

# design the portfolio
w = rpp.design(corr, b)
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
