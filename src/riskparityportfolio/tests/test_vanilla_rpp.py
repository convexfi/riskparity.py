import riskparityportfolio as rpp
import numpy as np
import pytest
import pdb


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


def test_random_covmat():
    N = 100
    b = np.ones(N)/N
    np.random.seed(42)
    #pdb.set_trace()
    U = np.random.multivariate_normal(mean=np.zeros(N), cov=0.1 * np.eye(N), size=round(.7 * N)).T
    Sigma = U @ U.T + np.eye(N)
    w = rpp.vanilla.design(Sigma, b, maxiter=1000, method="spinu")
    # assert that the portfolio respect the budget constraint
    np.testing.assert_almost_equal(np.sum(w), 1)
    # assert that the portfolio respect the no-shortselling constraint
    np.testing.assert_equal(all(w >= 0), True)
    # assert that the desired risk contributions are attained
    rc = w @ (Sigma * w)
    rc = rc / np.sum(rc)
    np.testing.assert_allclose(rc, b, atol=1/(10*N))


