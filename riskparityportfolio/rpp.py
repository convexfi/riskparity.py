import warnings
import tensorflow as tf
import numpy as np
from .sca import SuccessiveConvexOptimizer
from .riskfunctions import RiskContribOverBudgetDoubleIndex, RiskContribOverVarianceMinusBudget


__all__ = ['RiskParityPortfolio']


class RiskParityPortfolioValidator:
    def __init__(self):
        pass

class RiskParityPortfolio:
    """Designs risk parity portfolios by solving the following optimization problem

    minimize R(w) - alpha * mu.T * w + lambda * w.T Sigma w
    subject to Cw = c, Dw <= d

    where R is a risk concentration function, and alpha and beta are trade-off
    parameters for the expected return and the variance, respectively.

    Parameters
    ----------
    covariance : array, shape=(n, n)
        covariance matrix of the assets
    budget : array, shape=(n,)
        risk budget vector
    equality_constraints : string
        the equality constraint expression
    inequality_constraints : string
        the inequality constraint expression
    weights : array, shape=(n,)
        weights of the portfolio
    risk_concentration : class
        any valid child class of RiskConcentrationFunction
    """

    def __init__(self, covariance, budget=None,
                 equality_constraints=None,
                 inequality_constraints=None,
                 weights=None,
                 risk_concentration=None):
        self.covariance = covariance
        self.budget = budget
        self.weights = weights
        self.risk_concentration = risk_concentration
        self.validate()
        self.has_variance = False
        self.has_mean_return = False

    def get_diag_solution(self):
        w = np.sqrt(self.budget.numpy()) / np.sqrt(np.diagonal(self.covariance.numpy()))
        return w / w.sum()

    @property
    def mean_return(self):
        if self.has_mean_return:
            return tf.tensordot(self.weights, self.mean, axes=1)
        else:
            raise ValueError("the portfolio mean has not been specified, please use add_mean_return")

    @property
    def volatility(self):
        return tf.sqrt(tf.tensordot(self.weights,
                                    tf.linalg.matvec(self.covariance,
                                                     self.weights), axes=1))
    @property
    def risk_contributions(self):
        rc = tf.tensordot(self.weights, tf.multiply(self.covariance, self.weights), axes=1)
        return rc / tf.reduce_sum(rc)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if value is None:
            self._weights = tf.ones(self.number_of_assets, dtype=tf.float64) / self.number_of_assets
        else:
            try:
                self._weights = tf.convert_to_tensor(value)
            except Exception as e:
                raise e

    @property
    def number_of_assets(self):
        return self.covariance.shape[0]

    @property
    def risk_concentration(self):
        return self._risk_concentration

    @risk_concentration.setter
    def risk_concentration(self, value):
        if value is None:
            self._risk_concentration = RiskContribOverVarianceMinusBudget(self)
        elif isinstance(value, RiskConcetrationFunction):
            self._risk_concentration = value(self)
        else:
            raise ValueError("risk_concentration {} is not a valid child class"
                             "of RiskConcentrationFunction".format(value))

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value):
        if value is None:
            self._budget = tf.ones(self.number_of_assets, dtype=tf.float64) / self.number_of_assets
        else:
            try:
                self._budget = tf.convert_to_tensor(value)
            except Exception as e:
                raise e

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        if value.shape[0] != value.shape[1]:
            raise ValueError("shape mismatch: covariance matrix is not a square matrix")
        else:
            try:
                self._covariance = tf.convert_to_tensor(value)
            except Exception as e:
                raise e
            eigvals = np.linalg.eigvals(self._covariance.numpy())
            eigvals = np.sort(eigvals)
            if abs(eigvals[0] / eigvals[-1]) < 1e-6:
                warnings.warn("covariance matrix maybe singular")

    @property
    def equality_constraints(self):
        return self._equality_constraints

    @property
    def inequality_constraints(self):
        return self._inequality_constraints

    @property
    def box_constraints(self):
        return self._box_constraints

    def validate(self):
        if self.covariance.shape[0] != self.budget.shape[0]:
            raise ValueError("shape mismatch between covariance matrix and budget vector")

    def add_mean_return(self, alpha, mean):
        """Whether to consider the maximization of the mean return of the portfolio.

        Parameters
        ----------
        alpha : float
            Hyperparameter associated with the mean return
        mean : array, shape=(n,)
            Vector of means
        """
        self.alpha = alpha
        self.mean = mean
        self.has_mean_return = True

    def add_variance(self, lmd):
        """Whether to consider the minimization of the variance of the portfolio.

        Parameters
        ----------
        lmd : float
            Hyperparameter associated with the variance
        """
        self.lmd = lmd
        self.has_variance = True

    def design(self, **kwargs):
        """Optimize the portfolio.

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters to be passed to SuccessiveConvexOptimizer.
        """
        self.sca = SuccessiveConvexOptimizer(self, **kwargs)
        self.sca.solve()
