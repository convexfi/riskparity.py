import numpy as np
import riskparityportfolio as rpp
import pytest
import pdb

def test_on_tricky_example():
    """This is a test on a known somewhat tricky problem"""
    S = np.vstack(
        (
            np.array((1.0000, 0.0015, -0.0119)),
            np.array((0.0015, 1.0000, -0.0308)),
            np.array((-0.0119, -0.0308, 1.0000)),
        )
    )
    b = np.array((0.1594, 0.0126, 0.8280))
    ans = np.array([0.2798628, 0.08774909, 0.63238811])
    #pdb.set_trace()
    my_portfolio = rpp.RiskParityPortfolio(covariance=S, budget=b)
    my_portfolio.design()
    np.testing.assert_allclose(my_portfolio.weights, ans, rtol=1e-5)
    assert my_portfolio.risk_concentration.evaluate() < 1e-9


def test_random_covmat():
    N = 100
    b = np.ones(N)/N
    np.random.seed(42)
    U = np.random.multivariate_normal(mean=np.zeros(N), cov=0.1 * np.eye(N), size=round(.7 * N)).T
    Sigma = U @ U.T + np.eye(N)
    #pdb.set_trace()
    my_portfolio = rpp.RiskParityPortfolio(Sigma, budget=b)
    my_portfolio.design()
    w = my_portfolio.weights

    # assert that the portfolio respect the budget constraint
    np.testing.assert_almost_equal(np.sum(w), 1.0)
    # assert that the portfolio respect the no-shortselling constraint
    np.testing.assert_equal(all(w >= 0), True)
    # assert that the desired risk contributions are attained
    rc = w @ (Sigma * w)
    rc = rc / np.sum(rc)
    np.testing.assert_allclose(rc, b, atol=1/(10*N))


def test_singularity_issues_with_G_matrix():
    N = 100
    b = np.ones(N) / N
    np.random.seed(42)
    U = np.random.multivariate_normal(mean=np.zeros(N), cov=0.1 * np.eye(N), size=round(.7 * N)).T
    Sigma = U @ U.T  # singular covariance matrix

    my_portfolio = rpp.RiskParityPortfolio(Sigma, budget=b)
    with pytest.warns(UserWarning, match="Matrix Q is not positive definite: adding regularization term and then calling QP solver again."):
        my_portfolio.design(verbose=False, tau=1e-12)  # <-- force ill-conditioned matrix
    w_ref = my_portfolio.weights

    my_portfolio = rpp.RiskParityPortfolio(Sigma, budget=b)
    my_portfolio.design(tau=1e-4)  # <-- default tau, should be fine
    w = my_portfolio.weights
    np.testing.assert_allclose(w, w_ref, rtol=1e-3)

    # my_portfolio = rpp.RiskParityPortfolio(Sigma, budget=b)
    # my_portfolio.design(verbose=True, tau=0.05 * np.sum(np.diag(Sigma)) / (2 * N))
    # w = my_portfolio.weights
    # np.testing.assert_allclose(w, w_ref, rtol=1e-3)
    #
    # my_portfolio = rpp.RiskParityPortfolio(Sigma, budget=b)
    # my_portfolio.design(verbose=True, tau=2 * 0.1 ** 2 / (2 * N))
    # w = my_portfolio.weights
    # np.testing.assert_allclose(w, w_ref, rtol=1e-3)


def test_constraints():
    N = 50
    np.random.seed(42)
    U = np.random.multivariate_normal(mean=np.zeros(N), cov=0.1 * np.eye(N), size=round(.7 * N)).T
    Sigma = U @ U.T + np.eye(N)

    # Default constraints sum(w) = 1 and w >= 0:
    my_portfolio = rpp.RiskParityPortfolio(Sigma)
    my_portfolio.design()
    w1 = my_portfolio.weights
    np.testing.assert_almost_equal(np.sum(w1), 1)
    np.testing.assert_equal(all(w1 >= 0), True)

    # Equivalently, specifying explicitly sum(w) = 1:
    my_portfolio = rpp.RiskParityPortfolio(Sigma)
    my_portfolio.design(Cmat=np.ones((1, N)), cvec=np.array([1.0]))
    w2 = my_portfolio.weights
    np.testing.assert_allclose(w1, w2, rtol=1e-5)

    # Equivalently, specifying explicitly sum(w) = 1 and w >= 0:
    my_portfolio = rpp.RiskParityPortfolio(Sigma)
    my_portfolio.design(Cmat=np.ones((1, N)), cvec=np.array([1.0]),
                        Dmat=-np.eye(N), dvec=np.zeros(N))
    w3 = my_portfolio.weights
    np.testing.assert_allclose(w1, w3, rtol=1e-5)


    # Additional upper bound: w <= 0.03
    my_portfolio = rpp.RiskParityPortfolio(Sigma)
    my_portfolio.design(Cmat=np.ones((1, N)), cvec=np.array([1.0]),
                        Dmat=np.vstack([np.eye(N), -np.eye(N)]), dvec=np.concatenate([0.03*np.ones(N), np.zeros(N)]))
    w = my_portfolio.weights
    np.testing.assert_almost_equal(np.sum(w), 1)
    np.testing.assert_equal(all(w >= 0), True)
    print(max(w))
    np.testing.assert_array_less(w, (0.03 + 1e-3)*np.ones(N))


    # Bounds for sum(w): 0.5 <= sum(w) <= 1 (tending to upper bound)
    my_portfolio = rpp.RiskParityPortfolio(Sigma)
    my_portfolio.add_mean_return(alpha=1e-6, mean=np.ones(N))  # this adds sum(w) to also maximize sum(w)
    my_portfolio.design(Cmat=np.empty((0, N)), cvec=[],
                        Dmat=np.vstack([-np.ones((1,N)), np.ones((1,N))]), dvec=np.array([-0.5, 1]))
    w = my_portfolio.weights
    np.testing.assert_almost_equal(np.sum(w), 1.0)


    # Bounds for sum(w): 0.5 <= sum(w) <= 1 (tending to lower bound)
    my_portfolio = rpp.RiskParityPortfolio(Sigma)
    my_portfolio.add_mean_return(alpha=1e-6, mean=-np.ones(N))  # this adds -sum(w) to also minimize sum(w)
    my_portfolio.design(Cmat=np.empty((0, N)), cvec=[],
                        Dmat=np.vstack([-np.ones((1,N)), np.ones((1,N))]), dvec=np.array([-0.5, 1]))
    w = my_portfolio.weights
    np.testing.assert_almost_equal(np.sum(w), 0.5, decimal=3)



def test_dummy_variables():
    N = 50
    np.random.seed(42)
    U = np.random.multivariate_normal(mean=np.zeros(N), cov=0.1 * np.eye(N), size=round(.7 * N)).T
    Sigma = U @ U.T + np.eye(N)

    # Upper-bounded: sum(w) = 1 and 0 <= w <= 0.03
    my_portfolio = rpp.RiskParityPortfolio(Sigma)
    my_portfolio.design(Cmat=np.ones((1, N)), cvec=np.array([1.0]),
                        Dmat=np.vstack([np.eye(N), -np.eye(N)]), dvec=np.concatenate([0.03*np.ones(N), np.zeros(N)]))
    w_ref = my_portfolio.weights

    # Equivalently: sum(w) = 1, 0 <= w <= u, and u <= 0.03  (new dummy variable u, with w_tilde = [w; u])
    my_portfolio = rpp.RiskParityPortfolio(Sigma)
    my_portfolio.design(Cmat=np.hstack([np.ones((1, N)), np.zeros((1, N))]), cvec=np.array([1.0]),
                        Dmat=np.vstack([np.hstack([-np.eye(N), np.zeros((N, N))]),
                                        np.hstack([np.eye(N), -np.eye(N)]),
                                        np.hstack([np.zeros((N, N)), np.eye(N)])]),
                        dvec=np.concatenate([np.zeros(N), np.zeros(N), 0.03*np.ones(N)]))
    w = my_portfolio.weights

    np.testing.assert_allclose(w, w_ref, rtol=1e-5, atol=1e-5)
