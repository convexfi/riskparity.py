import numpy as np
import warnings
from tqdm import tqdm

try:
    import quadprog
except ImportError:
    import warnings

    warnings.warn(
        "not able to import quadprog." " the successive convex optimizer wont work."
    )

__all__ = ["SuccessiveConvexOptimizer"]


class SuccessiveConvexOptimizerValidator:
    def __init__(self):
        pass

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if not (0 < value < 1):
            raise ValueError(
                "gamma has to be a real number in the interval"
                "(0, 1), got {}.".format(value)
            )
        else:
            try:
                self._gamma = float(value)
            except Exception as e:
                raise e

    @property
    def zeta(self):
        return self._zeta

    @zeta.setter
    def zeta(self, value):
        if value < 0:
            raise ValueError(
                "zeta has to be a positive real number, got {}.".format(value)
            )
        else:
            try:
                self._zeta = float(value)
            except Exception as e:
                raise e

    @property
    def funtol(self):
        return self._funtol

    @funtol.setter
    def funtol(self, value):
        if value < 0:
            raise ValueError(
                "funtol has to be a positive real number, got {}.".format(value)
            )
        else:
            try:
                self._funtol = float(value)
            except Exception as e:
                raise e

    @property
    def wtol(self):
        return self._wtol

    @wtol.setter
    def wtol(self, value):
        if value < 0:
            raise ValueError(
                "wtol has to be a positive real number, got {}.".format(value)
            )
        else:
            try:
                self._wtol = float(value)
            except Exception as e:
                raise e

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        if value < 0:
            raise ValueError(
                "maxiter has to be a positive integer, got {}.".format(value)
            )
        else:
            try:
                self._maxiter = int(value)
            except Exception as e:
                raise e


class SuccessiveConvexOptimizer:
    """
    Successive Convex Approximation optimizer tailored for the risk parity problem including the linear constraints:
       Cmat @ w  = cvec
       Dmat @ w <= dvec,
    where matrices Cmat and Dmat have n columns (n being the number of assets). Based on the paper:

    Feng, Y., and Palomar, D. P. (2015). SCRIP: Successive convex optimization methods for risk parity portfolios design.
    IEEE Trans. Signal Processing, 63(19), 5285–5300.

    By default, the constraints are set to sum(w) = 1 and w >= 0, i.e.,
       Cmat = np.ones((1, n))
       cvec = np.array([1.0])
       Dmat = -np.eye(n)
       dvec = np.zeros(n).

    Notes:
        1) If equality constraints are not needed, set Cmat = np.empty((0, n)) and cvec = [].
        2) If the matrices Cmat and Dmat have more than n columns, it is assumed that the additional columns
        (same number for both matrices) correspond to dummy variables (which do not appear in the objective function).
    """
    def __init__(
        self,
        portfolio,
        tau=None,
        gamma=0.9,
        zeta=1e-7,
        funtol=1e-6,
        wtol=1e-6,
        maxiter=500,
        Cmat=None,
        cvec=None,
        Dmat=None,
        dvec=None,
    ):
        self.portfolio = portfolio
        self.tau = tau or 1e-4  # 0.05 * np.trace(self.portfolio.covariance) / (2 * self.portfolio.number_of_assets)
        sca_validator = SuccessiveConvexOptimizerValidator()
        self.gamma = sca_validator.gamma = gamma
        self.zeta = sca_validator.zeta = zeta
        self.funtol = sca_validator.funtol = funtol
        self.wtol = sca_validator.wtol = wtol
        self.maxiter = sca_validator.maxiter = maxiter
        self.Cmat = Cmat
        self.Dmat = Dmat  # Dmat @ w <= dvec
        self.cvec = cvec
        self.dvec = dvec
        self.number_of_vars = self.Cmat.shape[1]
        self.number_of_dummy_vars = self.number_of_vars - self.portfolio.number_of_assets
        self.dummy_vars = np.zeros(self.number_of_dummy_vars)
        self.CCmat = np.vstack((self.Cmat, -self.Dmat)).T  # CCmat.T @ w >= bvec
        self.bvec = np.concatenate((self.cvec, -self.dvec))
        self.meq = self.Cmat.shape[0]
        self._funk = self.get_objective_function_value()
        self.objective_function = [self._funk]
        self._tauI = self.tau * np.eye(self.portfolio.number_of_assets)
        self.Amat = self.portfolio.risk_concentration.jacobian_risk_concentration_vector()
        self.gvec = self.portfolio.risk_concentration.risk_concentration_vector

    @property
    def Cmat(self):
        return self._Cmat

    @Cmat.setter
    def Cmat(self, value):
        if value is None:
            self._Cmat = np.atleast_2d(np.ones(self.portfolio.number_of_assets))
        elif np.atleast_2d(value).shape[1] >= self.portfolio.number_of_assets:
            self._Cmat = np.atleast_2d(value)
        else:
            raise ValueError(
                "Cmat shape {} doesnt agree with the number of"
                "assets {}".format(value.shape, self.number_of_assets)
            )

    @property
    def Dmat(self):
        return self._Dmat

    @Dmat.setter
    def Dmat(self, value):
        if value is None:
            self._Dmat = -np.eye(self.portfolio.number_of_assets)
        elif np.atleast_2d(value).shape[1] == self.Cmat.shape[1]:
            self._Dmat = np.atleast_2d(value)
        else:
            raise ValueError(
                "Dmat shape {} doesnt agree with the number of"
                "assets {}".format(value.shape, self.number_of_assets)
            )

    @property
    def cvec(self):
        return self._cvec

    @cvec.setter
    def cvec(self, value):
        if value is None:
            self._cvec = np.array([1.0])
        elif len(value) == self.Cmat.shape[0]:
            self._cvec = value
        else:
            raise ValueError(
                "cvec shape {} doesnt agree with Cmat shape"
                "{}".format(value.shape, self.Cmat.shape)
            )

    @property
    def dvec(self):
        return self._dvec

    @dvec.setter
    def dvec(self, value):
        if value is None:
            self._dvec = np.zeros(self.portfolio.number_of_assets)
        elif len(value) == self.Dmat.shape[0]:
            self._dvec = np.atleast_1d(value)
        else:
            raise ValueError(
                "dvec shape {} doesnt agree with Dmat shape"
                "{}".format(value.shape, self.Dmat.shape)
            )

    def get_objective_function_value(self):
        obj = self.portfolio.risk_concentration.evaluate()
        if self.portfolio.has_mean_return:
            obj -= self.portfolio.alpha * self.portfolio.mean_return
        if self.portfolio.has_variance:
            obj += self.portfolio.lmd * self.portfolio.volatility ** 2
        return obj

    def iterate(self, verbose=True):
        wk = self.portfolio.weights
        g = self.gvec(wk)
        A = np.ascontiguousarray(self.Amat(wk))
        At = np.transpose(A)
        Q = 2 * At @ A + self._tauI
        q = 2 * np.matmul(At, g) - Q @ wk  # np.matmul() is necessary here since g is not a numpy array
        if self.portfolio.has_variance:
            Q += self.portfolio.lmd * self.portfolio.covariance
        if self.portfolio.has_mean_return:
            q -= self.portfolio.alpha * self.portfolio.mean
        if self.number_of_dummy_vars > 0:
            Q = np.vstack([np.hstack([Q, np.zeros((self.portfolio.number_of_assets, self.number_of_dummy_vars))]),
                           np.hstack([np.zeros((self.number_of_dummy_vars, self.portfolio.number_of_assets)),
                                      self.tau * np.eye(self.portfolio.number_of_assets)])])
            q = np.concatenate([q, -self.tau * self.dummy_vars])
        # Call QP solver (min 0.5*x.T G x + a.T x  s.t.  C.T x >= b) controlling for ill-conditioning:
        try:
            w_hat = quadprog.solve_qp(Q, -q, C=self.CCmat, b=self.bvec, meq=self.meq)[0]
        except ValueError as e:
            if str(e) == "matrix G is not positive definite":
                warnings.warn(
                    "Matrix Q is not positive definite: adding regularization term and then calling QP solver again.")
                # eigvals = np.linalg.eigvals(Q)
                # print("    - before regularization: cond. number = {:,.0f}".format(max(eigvals) / min(eigvals)))
                # print("    - after regularization: cond. number = {:,.0f}".format(max(eigvals + np.trace(Q)/1e7) / min(eigvals + np.trace(Q)/1e7)))
                Q += np.eye(Q.shape[0]) * np.trace(Q)/1e7
                w_hat = quadprog.solve_qp(Q, -q, C=self.CCmat, b=self.bvec, meq=self.meq)[0]
            else:
                # If the error is different, re-raise it
                raise
        self.portfolio.weights = wk + self.gamma * (w_hat[:self.portfolio.number_of_assets] - wk)
        fun_next = self.get_objective_function_value()
        self.objective_function.append(fun_next)
        has_w_converged = (
                (np.abs(self.portfolio.weights - wk) <= self.wtol * 0.5 * (np.abs(self.portfolio.weights) + np.abs(wk)))
                | ((np.abs(self.portfolio.weights) < 1e-6) & (np.abs(wk) < 1e-6))
        ).all()
        has_fun_converged = (
                (np.abs(self._funk - fun_next) <= self.funtol * 0.5 * (np.abs(self._funk) + np.abs(fun_next)))
                | ((np.abs(self._funk) <= 1e-10) & (np.abs(fun_next) <= 1e-10))
        )
        if self.number_of_dummy_vars > 0:
            have_dummies_converged = (
                    (np.abs(w_hat[self.portfolio.number_of_assets:] - self.dummy_vars) <= self.wtol * 0.5 *
                     (np.abs(w_hat[self.portfolio.number_of_assets:]) + np.abs(self.dummy_vars)))
                    | ((np.abs(w_hat[self.portfolio.number_of_assets:]) < 1e-6) & (np.abs(self.dummy_vars) < 1e-6))
            ).all()
            self.dummy_vars = w_hat[self.portfolio.number_of_assets:]
        else:
            have_dummies_converged = True
        if (has_w_converged and have_dummies_converged) or has_fun_converged:
            # if verbose:
            #     print(f"  Has func. converged: {has_fun_converged}; has w converged: {has_w_converged}")
            return False
        self.gamma = self.gamma * (1 - self.zeta * self.gamma)
        self._funk = fun_next
        return True

    def solve(self, verbose=True):
        i = 0
        iterator = range(self.maxiter)
        if verbose:
            iterator = tqdm(iterator)
        for _ in iterator:
            if not self.iterate(verbose=verbose):
                break
            i += 1

def project_line_and_box(weights, lower_bound, upper_bound):
    def objective_function(variable, weights):
        pass

    pass
