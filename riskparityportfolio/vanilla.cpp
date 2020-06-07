#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace Eigen;
using namespace std;

// Cyclical coordinate descent for Spinu's formulation
// of the risk parity portfolio problem
Eigen::VectorXd risk_parity_portfolio_ccd_spinu(const Eigen::MatrixXd& Sigma,
                                                const Eigen::VectorXd& b,
                                                const double tol = 1E-8,
                                                const unsigned int maxiter = 50) {
  double aux, x_diff, xk_sum;
  const unsigned int n = b.size();
  Eigen::VectorXd xk = (1 / Sigma.diagonal().array().sqrt()).matrix();
  Eigen::VectorXd x_star(n), Sigma_xk(n), rc(n);
  Sigma_xk = Sigma * xk;
  for (unsigned int k = 0; k < maxiter; ++k) {
    for (unsigned int i = 0; i < n; ++i) {
      // compute update for the portfolio weights x
      aux = xk(i) * Sigma(i, i) - Sigma_xk(i);
      x_star(i) = (.5 / Sigma(i, i)) * (aux + std::sqrt(aux * aux + 4 * Sigma(i, i) * b(i)));
      // update auxiliary terms
      x_diff = x_star(i) - xk(i);
      Sigma_xk += (Sigma.col(i).array() * x_diff).matrix();
      xk(i) = x_star(i);
    }
    xk_sum = xk.sum();
    rc = (xk.array() * (Sigma_xk).array() / (xk_sum * xk_sum)).matrix();
    if ((rc.array() / rc.sum() - b.array()).abs().maxCoeff() < tol)
      break;
  }
  return x_star / xk_sum;
}


namespace py = pybind11;


PYBIND11_MODULE(vanilla, m) {
  m.doc() = "design of risk parity portfolios";
  m.def("design", &risk_parity_portfolio_ccd_spinu,
        py::arg("Sigma"), py::arg("b"),
        py::arg("tol") = 1E-8, py::arg("maxiter") = 50,
        R"pbdoc(
          A function to design vanilla risk parity (budgeting) portfolios.

          This is an implementation of the risk formulation proposed by Spinu
          (2013). The algorithm was inspired by the cyclical coordinate descent
          proposed by Griveau-Billion (2013).

          Parameters
          ----------
          Sigma : numpy.ndarray
            n x n covariance matrix of the assets

          b : numpy.ndarray
            n x 1 risk budgeting vector

          tol : float
            tolerance on the risk contributions convergence. Default value is 1e-8

          maxiter : int
            maximum number of iterations. Default value is 50


          Example
          -------
          >>> import numpy as np
          >>> import riskparityportfolio as rpp
          >>> np.random.seed(42)

          # creates a correlation matrix from time-series of five assets
          >>> x = np.random.normal(size=1000).reshape((5, -1))
          >>> corr = x @ x.T

          # create the desired risk budgeting vector
          >>> b = np.ones(len(corr)) / len(corr)

          # design the portfolio
          >>> w = rpp.design(corr, b)
          >>> w
          array([ 0.21075375,  0.21402865,  0.20205399,  0.16994639,  0.20321721])

          # compute the risk budgeting
          >>> rc = w @ (corr * w)
          >>> rc / np.sum(rc)
          array([ 0.2,  0.2,  0.2,  0.2,  0.2])

          # let's try a different budget
          >>> b = np.array([0.01, 0.09, .1, .1, .7])
          >>> w = rpp.design(corr, b)
          >>> w
          array([ 0.06178354,  0.19655744,  0.16217134,  0.12808275,  0.45140493])
          >>> rc = w @ (corr * w)
          >>> rc / np.sum(rc)
          array([ 0.01,  0.09,  0.1 ,  0.1 ,  0.7 ])


          References
          ----------
          [1] F. Spinu (2013). An Algorithm for Computing Risk Parity Weights. https://dx.doi.org/10.2139/ssrn.2297383
          [2] T. Griveau-Billion et. al. (2013). A fast algorithm for computing High-dimensional risk parity portfolios.
              https://arxiv.org/pdf/1311.4057.pdf


          Notes
          -----
          To get the risk parity portfolio, set `b` as the 1/n uniform vector.
        )pbdoc"
       );

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
