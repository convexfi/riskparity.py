import riskparityportfolio as rpp
import numpy as np

def test():
    n = 1000
    t = 1000 * n
    x = np.random.normal(size=t).reshape((n, -1))
    corr = x @ x.T / t
    b = np.ones(len(corr)) / len(corr)
    w = rpp.vanilla.design(corr, b)
    rc = w @ (corr * w)
    rc = rc / np.sum(rc)
    # assert that the portfolio respect the budget constraint
    np.testing.assert_almost_equal(np.sum(w), 1)
    # assert that the portfolio repect the no-shortselling constraint
    np.testing.assert_equal(all(w > 0), True)
    # assert that the desired risk contributions are attained
    np.testing.assert_allclose(rc, b, rtol = 1e-4)
