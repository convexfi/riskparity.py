import numpy as np
from ..rpp import RiskParityPortfolio

def test_on_tricky_example():
    """ This is a test on a known somewhat tricky problem
    """
    S = np.vstack((np.array((1.0000, 0.0015, -0.0119)),
                   np.array((0.0015, 1.0000, -0.0308)),
                   np.array((-0.0119, -0.0308, 1.0000))))
    b = np.array((0.1594, 0.0126, 0.8280))
    ans = np.array([0.2798628 , 0.08774909, 0.63238811])
    rpp = RiskParityPortfolio(covariance=S, budget=b)
    rpp.design()
    np.testing.assert_allclose(rpp.weights, ans, rtol = 1e-5)
    assert rpp.risk_concentration.evaluate() < 1e-9
