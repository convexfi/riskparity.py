# riskparity.py

[![PyPI version](https://badge.fury.io/py/riskparityportfolio.svg)](https://badge.fury.io/py/riskparityportfolio)
[![Downloads](https://pepy.tech/badge/riskparityportfolio)](https://pepy.tech/project/riskparityportfolio)
[![codecov](https://codecov.io/gh/mirca/riskparity.py/branch/master/graph/badge.svg)](https://codecov.io/gh/mirca/riskparity.py)


**riskparityportfolio** provides solvers to design risk parity portfolios.
In its simplest form, we consider the convex formulation with a unique solution proposed by
[Spinu (2013)](https://dx.doi.org/10.2139/ssrn.2297383) and use cyclical methods inspired by
[Griveau-Billion et al. (2013)](https://arxiv.org/pdf/1311.4057.pdf)
and [Choi & Chen (2022)](https://www.emerald.com/insight/content/doi/10.1108/JDQS-12-2021-0031/full/pdf). For more general formulations,
which are usually nonconvex, we implement the successive convex approximation
method proposed by [Feng & Palomar (2015)](https://doi.org/10.1109/TSP.2015.2452219).

**Documentation:** [**https://mirca.github.io/riskparity.py**](https://mirca.github.io/riskparity.py)

**R version:** [**https://mirca.github.io/riskParityPortfolio**](https://mirca.github.io/riskParityPortfolio)

**Rust version:** [**https://github.com/mirca/riskparity.rs**](https://github.com/mirca/riskparity.rs)

**Talks**: [**slides HKML meetup 2020**](https://speakerdeck.com/mirca/breaking-down-risk-parity-portfolios-a-practical-open-source-implementation),
[**tutorial - Data-driven Portfolio Optimization Course (HKUST)**](https://www.youtube.com/watch?v=xb1Xxf5LQks)

## Installation

* **development version**

```
$ git clone https://github.com/dppalomar/riskparity.py.git
$ cd riskparity.py
$ pip install -e .
```

* **stable version**

```
$ pip install riskparityportfolio
```

### Windows requirements

Make sure to install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
prior to ``riskparityportfolio``.

``riskparityportfolio`` depends on ``jaxlib`` which can be installed following these
[instructions](https://github.com/cloudhan/jax-windows-builder).


## References

* Spinu, Florin. An Algorithm for Computing Risk Parity Weights (July 30, 2013). Available at SSRN: [https://ssrn.com/abstract=2297383](https://ssrn.com/abstract=2297383).

* Griveau-Billion, Théophile et al. A Fast Algorithm for Computing High-dimensional Risk Parity Portfolios. [https://arxiv.org/abs/1311.4057](https://arxiv.org/abs/1311.4057)

* Feng, Yiyong et al. SCRIP: Successive Convex Optimization Methods for Risk Parity Portfolio Design.
IEEE Transactions on Signal Processing, 2015. [https://ieeexplore.ieee.org/document/7145485](https://ieeexplore.ieee.org/document/7145485)

* Choi, J., & Chen, R. (2022). Improved iterative methods for solving risk parity portfolio. Journal of Derivatives and Quantitative Studies 30(2), 114–124. [https://doi.org/10.1108/JDQS-12-2021-0031](https://doi.org/10.1108/JDQS-12-2021-0031)


## License

Copyright 2022 [Ze Vinicius](https://mirca.github.io) and [Daniel Palomar](https://www.danielppalomar.com)

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
