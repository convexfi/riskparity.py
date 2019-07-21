import warnings
import tensorflow as tf
import numpy as np
import osqp
from scipy.sparse import csc_matrix
from .sca import SuccessiveConvexOptimizer
from .riskfunctions import RiskContribOverBudgetDoubleIndex


__all__ = ['RiskParityPortfolio']


class RiskParityPortfolioValidator:
    def __init__(self):
        pass

class RiskParityPortfolio:

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
        self._risk_concentration = RiskContribOverBudgetDoubleIndex(weights=self.weights,
                                                                    covariance=self.covariance,
                                                                    budget=self.budget)

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value):
        if value is None:
            self._budget = tf.ones(self.number_of_assets) / self.number_of_assets
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

    def design(self, **kwargs):
        sca = SuccessiveConvexOptimizer(self, **kwargs)
        sca.solve()
