#ifndef VANILLA_H
#define VANILLA_H
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef const Eigen::Matrix<double, Eigen::Dynamic, 1> c_vector_t;
typedef const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> c_matrix_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

// Cyclical coordinate descent for Spinu's formulation
// of the risk parity portfolio problem
vector_t risk_parity_portfolio_ccd_spinu(c_matrix_t& Sigma,
                                         c_vector_t& b,
                                         const double tol = 1E-4,
                                         const int maxiter = 100) {
  double aux, x_diff, xk_sum;
  auto n = b.size();
  vector_t xk = vector_t::Constant(n, 1);
  xk = std::sqrt(1.0 / Sigma.sum()) * xk;
  vector_t x_star(n), Sigma_xk(n), rc(n);
  Sigma_xk = Sigma * xk;
  for (auto k = 0; k < maxiter; ++k) {
    for (auto i = 0; i < n; ++i) {
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


// cyclical coordinate descent algo by Choi & Chen 2022
// ref: https://arxiv.org/pdf/2203.00148.pdf
vector_t risk_parity_portfolio_ccd_choi(c_matrix_t& cov,
                                        c_vector_t& b,
                                        const double tol = 1E-4,
                                        const int maxiter = 100) 
  {
  double ai;
  auto n = b.size();
  vector_t vol = cov.diagonal().array().sqrt();
  vector_t invvol = (1 / vol.array()).matrix();
  matrix_t corr = cov.array().colwise() * invvol.array();
  corr = corr.array().rowwise() * invvol.transpose().array();
  matrix_t adj = corr;
  adj.diagonal().array() = 0;
  vector_t wk = vector_t::Ones(n);
  wk = (wk.array() / std::sqrt(corr.sum())).matrix();
  for (auto k = 0; k < maxiter; ++k) {
    for (auto i = 0; i < n; ++i) {
      // compute portfolio weights
      ai = 0.5 * ((adj.col(i).array() * wk.array()).sum());
      wk(i) = std::sqrt(ai * ai + b(i)) - ai;
    }
    wk = wk.array() / std::sqrt(wk.transpose() * corr * wk);
    if ((wk.array() * (corr * wk).array() - b.array()).abs().maxCoeff() < tol)
      break;
  }
  vector_t w = wk.array() / vol.array();
  return (w / w.sum()).matrix();
}

#endif


