import numpy as np

class RiskParityPortfolio:

    def __init__(self, covariance, budget=None,
                 equality_constraints=None, inequality_constraints=None):
        self.covariance = covariance
        self.number_of_assets = self.covariance.shape[0]
        self.budget = budget
        self.validate_problem()

    @property
    def budget(self):
        return self._budget

    @buget.setter
    def budget(self, value):
        if value is None:
            self._budget = np.ones(self.number_of_assets) / self.number_of_assets
        else:
            try:
                self._budget = np.asarray(value)
            except e:
                raise e

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        if value.shape[0] != value.shape[1]:
            raise ValueError("shape mismatch: covariance matrix is not a square matrix")
        else:
            eigvals = np.linalg.eigvals(value)
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

    def validate_problem(self):
        if self.covariance.shape[0] != self.budget.shape[0]:
            raise ValueError("shape mismatch between covariance matrix and budget vector")


def project_line_and_box(weights, lower_bound, upper_bound):
    def objective_function(variable, weights):
