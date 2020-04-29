.. riskparity.py documentation master file, created by
   sphinx-quickstart on Tue Jul 23 09:34:53 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

riskparity.py
=============

**riskparity.py** implements fast and scalable algorithms to design risk (budgeting) parity portfolios.
The algorithms are based on the works of Spinu (2013), Griveau-Billion *et. al.* (2013), and Feng & Palomar (2015).

We consider the following optimization problem and its particular cases

.. math::

        \begin{array}{ll}
        \underset{\mathbf{w}}{\textsf{minimize}} &
        R(\mathbf{w}) - \alpha \mathbf{w}^{\top}\boldsymbol{\mu} + \lambda \mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}\\
        \textsf{subject to} & \mathbf{C}\mathbf{w} = \mathbf{c}, \mathbf{D}\mathbf{w} \leq \mathbf{d}
        \end{array}

where :math:`R` is a risk concentration function.

Installation
------------

**riskparity.py** can be installed via pip as::

$ pip install riskparityportfolio

Its *development* version can be installed as::

$ git clone https://github.com/dppalomar/riskparity.py
$ cd riskparity.py
$ pip install -e .

Dependencies
------------

**riskparity.py** is built on top of **numpy**, **jax**, **quadprog**, **pybind**, and **tqdm**.

R Version
---------

An R version of this package is available at https://github.com/dppalomar/riskParityPortfolio

Tutorials
---------

.. toctree::

        tutorials/minimal-usage.ipynb
        tutorials/including-mean-return-and-variance.ipynb
        tutorials/comparison-with-pyrb.ipynb

References
----------

* F. Spinu, "An algorithm for computing risk parity weights", SSRN, 2013.
* T. Griveau-Billion, J. Richard, and T. Roncalli, "A fast algorithm for computing high-dimensional risk parity portfolios" ArXiv preprint, 2013.
* Y. Feng and D. P. Palomar, "SCRIP: Successive convex optimization methods for risk parity portfolios design" IEEE Trans. Signal Process., vol. 63, no. 19, pp. 5285–5300, Oct. 2015.
* J. Richard, and T. Roncalli "Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles" (February 8, 2019). Available at SSRN: https://ssrn.com/abstract=3331184

.. toctree::
   :maxdepth: 2
   :caption: Contents:

