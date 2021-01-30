# riskparity.py

[![PyPI version](https://badge.fury.io/py/riskparityportfolio.svg)](https://badge.fury.io/py/riskparityportfolio)
[![Downloads](https://pepy.tech/badge/riskparityportfolio)](https://pepy.tech/project/riskparityportfolio)
[![codecov](https://codecov.io/gh/mirca/riskparity.py/branch/master/graph/badge.svg)](https://codecov.io/gh/mirca/riskparity.py)


**riskparityportfolio** provides tools to design risk parity portfolios.
In its simplest form, we consider the convex formulation with a unique solution proposed by
[Spinu (2013)](https://dx.doi.org/10.2139/ssrn.2297383) and use a cyclical method inspired by
[Griveau-Billion (2013)](https://arxiv.org/pdf/1311.4057.pdf). For more general formulations,
which are usually nonconvex, we implement the successive convex approximation
method proposed by [Feng & Palomar (2015)](https://doi.org/10.1109/TSP.2015.2452219).

**Documentation:** [**https://mirca.github.io/riskparity.py**](https://mirca.github.io/riskparity.py)

**R version:** [**https://mirca.github.io/riskParityPortfolio**](https://mirca.github.io/riskParityPortfolio)

**Talks**: [**slides HKML meetup 2020**](https://speakerdeck.com/mirca/breaking-down-risk-parity-portfolios-a-practical-open-source-implementation),
[**tutorial - Data-driven Portfolio Optimization Course (HKUST)**](https://www.youtube.com/watch?v=xb1Xxf5LQks)

## Installation

```
$ git clone https://github.com/dppalomar/riskparity.py.git
$ cd riskparity.py
$ pip install -e .
```


## License

Copyright 2019 Ze Vinicius and Daniel Palomar

This project is licensed under the terms of the MIT License.

## Disclaimer

The information, software, and any additional resources contained in this repository are not intended as,
and shall not be understood or construed as, financial advice. Past performance is not a reliable indicator
of future results and investors may not recover the full amount invested.
The [authors](https://github.com/dppalomar/riskParityPortfolio/blob/master/AUTHORS.md) of this repository
accept no liability whatsoever for any loss or damage you may incur.  Any opinions expressed in this repository
are from the personal research and experience of the
[authors](https://github.com/dppalomar/riskParityPortfolio/blob/master/AUTHORS.md) and are intended as
educational material.
