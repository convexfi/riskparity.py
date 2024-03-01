import riskparityportfolio as rpp
import numpy as np
import pytest

@pytest.mark.parametrize("method", ["choi", "spinu"])
def test(method):
    n = 100
    t = 1000 * n
    x = np.random.normal(size=t).reshape((n, -1))
    cov = np.cov(x)
    b = np.random.uniform(size=n)
    b = b / np.sum(b)
    w = rpp.vanilla.design(cov, b, maxiter=1000, method=method)
    rc = w @ (cov * w)
    rc = rc / np.sum(rc)
    # assert that the portfolio respect the budget constraint
    np.testing.assert_almost_equal(np.sum(w), 1)
    # assert that the portfolio respect the no-shortselling constraint
    np.testing.assert_equal(all(w >= 0), True)
    # assert that the desired risk contributions are attained
    np.testing.assert_allclose(rc, b, atol=1/(10*n))
